# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from lfwa_eval import voc_eval, roc_curve_single_attribute, dataset_eval
from fast_rcnn.config import cfg
from matplotlib import pyplot as plt

class lfwa(imdb):
    def __init__(self, image_set, devkit_path=None):
        imdb.__init__(self, 'lfwa_' + image_set)
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'LFWA')
        self._classes = ('__background__', # always index 0
                         'eye', 'nose', 'mouth', 'upper', 'lower', 'face')
        self._face_attributes_name = ('5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive' ,'Bags_Under_Eyes',
                                      'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
                                      'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
                                      'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
                                      'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                                      'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes',
                                      'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
                                      'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling',
                                      'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
                                      'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young' )
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_name()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        # min_size here means the minimum size of the boxes to keep
        # cleanup means whether to clean up the voc results file or not
        self.config = {'cleanup'     : False,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._devkit_path), \
                'LfwAdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image index in the image sequence.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  self._image_index[i])
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path


    def image_path_from_image_name(self, image_name):
        """
        Construct an image path from the image's image name.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  image_name)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_name(self):
        """
        Load the image names listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /CelebAdevkit/CelebA/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_names = [x.strip() for x in f.readlines()]
        return image_names

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'CelebAdevkit')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_lfwa_annotation(image_name)
                    for image_name in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def rpn_roidb(self):
        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_lfwa_annotation(self, image_name):
        """
        Load image and bounding boxes info and face attributes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', image_name[0:-4] + '.xml')
        tree = ET.parse(filename)
        # load the face attributes info
        attributes = tree.findall('attribute')
        num_attributes = len(attributes)
        face_attrs = np.zeros((num_attributes), dtype=np.int32);

        for ix, attribute in enumerate(attributes):
            face_attrs[ix] = int(attribute.find('value').text)

        # load the boxes info
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas,
                'face_attrs': face_attrs}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def _get_lfwa_results_file_template(self):
        # CelebAdevkit/results/CelebA/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_attr_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            'LfwA',
            'Main',
            filename)
        return path

    def _write_lfwa_results_file(self, all_probs):
        for attr_ind, attr in enumerate(self._face_attributes_name):
            print 'Writing {} LfwA results file'.format(attr)
            filename = self._get_lfwa_results_file_template().format(attr)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    attr_prob = all_probs[im_ind]
                    f.write('{:s} {:.3f} {:.3f}\n'.
                            format(index, attr_prob[attr_ind, 0],
                                   attr_prob[attr_ind, 1]))

    def _do_python_eval(self, output_dir = 'output'):
        annopath = os.path.join(
            self._devkit_path,
            'LFWA',
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'LFWA',
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        # use_07_metric = True if int(self._year) < 2010 else False
        use_07_metric = True
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, attr in enumerate(self._face_attributes_name):
            filename = self._get_lfwa_results_file_template().format(attr)
            rec, prec, acc = voc_eval(
                filename, annopath, imagesetfile, attr, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            ap = roc_curve_single_attribute(filename, annopath, imagesetfile, attr, cachedir)
            aps += [ap]
            print('recall for {} = {:.4f}'.format(attr, rec))
            print('precision for {} = {:.4f}'.format(attr, prec))
            print('accuracy for {} = {:.4f}'.format(attr, acc))
            print('average precision for {} = {:.4f}'.format(attr, ap))
            with open(os.path.join(output_dir, attr + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print '-----------------------------------------------------'
        print 'Computing results with the official MATLAB eval code.'
        print '-----------------------------------------------------'
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
               .format(self._devkit_path, self._get_comp_id(),
                       self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def dataset_eval(self):
        annopath = os.path.join(
            self._devkit_path,
            'LFWA',
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'LFWA',
            'ImageSets',
            'Main',
            'train.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        ratio_positive_array = np.zeros(len(self._face_attributes_name))

        for i, attr in enumerate(self._face_attributes_name):
            num_images, ratio_positive = dataset_eval(annopath, imagesetfile,
                                          attr, cachedir)
            ratio_positive_array[i] = ratio_positive
            print('number of samples for {} = {:.4f}'.format(attr, num_images))
            print('positive sample ratio for {} = {:.4f}'.format(attr, ratio_positive))

        # bar plot
        fig = plt.figure()
        # get subplot, 111 means split the figure into 1 (rows) * 1 (ncols) sub-axes
        ax = fig.add_subplot(111)

        # bar width
        width = 0.35
        left = np.arange(len(self._face_attributes_name))

        # bars
        rects1 = ax.bar(left, ratio_positive_array, width,
                        color = 'red',
                        align = 'center')

        # axes and labels
        ax.set_xlim(-width, len(left) + width)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Ratio of positive samples')
        ax.set_title('Ratio of positive samples for each category of attributes')
        ax.set_xticks(left)
        ax.set_xticklabels(self._face_attributes_name, rotation=90)

        # add a legend
        ax.legend((rects1[0], ), ('LFWA',))
        plt.show()

    def evaluate_attributes(self, all_probs, output_dir):
        self._write_lfwa_results_file(all_probs)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for attr in self._face_attributes_name:
                filename = self._get_lfwa_results_file_template().format(attr)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.lfwa import lfwa
    d = lfwa('trainval')
    res = d.roidb
    from IPython import embed; embed()
