#include <cfloat>

#include "caffe/plain_zf_layers.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void GRIDPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  GRIDPoolingParameter grid_pool_param = this->layer_param_.grid_pooling_param();
  CHECK_GT(grid_pool_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(grid_pool_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = grid_pool_param.pooled_h();
  pooled_width_ = grid_pool_param.pooled_w();
}

template <typename Dtype>
void GRIDPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
}

template <typename Dtype>
void GRIDPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  int* mask = max_idx_.mutable_cpu_data();
  caffe_set(top_count, -1, mask);
  
  // bin size 
  const Dtype bin_size_h = static_cast<Dtype>(height_) / static_cast<Dtype>(pooled_height_);
  const Dtype bin_size_w = static_cast<Dtype>(width_) / static_cast<Dtype>(pooled_width_);

  for (int n = 0; n < bottom[0]->num(); ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          //  start (included) = floor(ph * height_ / pooled_height_)
          //  end (excluded) = ceil((ph + 1) * height_ / pooled_height_)
          int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
            * bin_size_h));
          int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
            * bin_size_w));
          int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
            * bin_size_h));
          int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
           * bin_size_w));

          hend = min(max(hend,0), height_);
          wend = min(max(wend,0), width_);
          hstart = min(max(hstart,0), height_);
          wstart = min(max(wstart, 0), width_);
          const int pool_index = ph * pooled_width_ + pw;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int index = h * width_ + w;
              if (bottom_data[index] > top_data[pool_index]) {
                top_data[pool_index] = bottom_data[index];
                mask[pool_index] = index;
              }
            }
          }
        }
      }

      // compute offset
      bottom_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
      mask += top[0]->offset(0, 1);
    }
  }
}  

template <typename Dtype>
void GRIDPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(GRIDPoolingLayer);
#endif

INSTANTIATE_CLASS(GRIDPoolingLayer);
REGISTER_LAYER_CLASS(GRIDPooling);

}  // namespace caffe
