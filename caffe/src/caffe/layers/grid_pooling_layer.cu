#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/plain_zf_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void GRIDPoolForward(int nthreads,
    const Dtype* const bottom_data,  int num,  int channels,
     int height,  int width,  int pooled_height,
     int pooled_width, Dtype* top_data, int* mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    // index = (n * channels * pooled_height * pooled_width) + (c * pooled_height * pooled_width) + (pooled_width * ph) + pw
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    // bin size
    Dtype bin_size_h = static_cast<Dtype>(height)
      / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(width)
      / static_cast<Dtype>(pooled_width);
    // hstart, wstart, hend, wend
    int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
                                        * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
                                        * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
                                     * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
                                     * bin_size_w));

    hstart = min(max(hstart, 0), height);
    wstart = min(max(wstart, 0), width);
    hend = min(max(hend,0), height);
    wend = min(max(wend,0), width);

    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_slice[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_slice[maxidx];
        }
      }
    }
    top_data[index] = maxval;
    mask[index] = maxidx;
  }
}

template <typename Dtype>
void GRIDPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int* mask = max_idx_.mutable_gpu_data();
  int count = top[0]->count();
  
  GRIDPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bottom[0]->num(), channels_,
      height_, width_, pooled_height_, pooled_width_, top_data,
      mask);

  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void GRIDPoolBackward(const int nthreads, const Dtype* const top_diff,
    const int* const mask, const int num, const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;

    Dtype bin_size_h = static_cast<Dtype>(height)
                         / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(width)
                         / static_cast<Dtype>(pooled_width);

    int phstart = floor(static_cast<Dtype>(h) / bin_size_h);
    int phend = ceil(static_cast<Dtype>(h + 1) / bin_size_h);
    int pwstart = floor(static_cast<Dtype>(w) / bin_size_w);
    int pwend = ceil(static_cast<Dtype>(w + 1) / bin_size_w);

    phstart = min(max(phstart, 0), pooled_height);
    phend = min(max(phend, 0), pooled_height);
    pwstart = min(max(pwstart, 0), pooled_width);
    pwend = min(max(pwend, 0), pooled_width);

    Dtype gradient = 0;
    const int offset = (n * channels + c) * pooled_height * pooled_width;
    const Dtype* const top_diff_slice = top_diff + offset;
    
    const int* const mask_slice = mask + offset;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        if (mask_slice[ph * pooled_width + pw] == h * width + w) {
          gradient += top_diff_slice[ph * pooled_width + pw];
        }
      }
    }
    
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void GRIDPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  const int* mask = max_idx_.gpu_data();

  // NOLINT_NEXT_LINE(whitespace/operators)
  GRIDPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_,
        bottom_diff);

  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(GRIDPoolingLayer);


}  // namespace caffe
