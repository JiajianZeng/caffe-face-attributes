# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from fast_rcnn.nms_wrapper import nms
import cPickle
from utils.blob import im_list_to_blob
import os
from PIL import Image, ImageDraw
from utils.vis import vis_square

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois':None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    if not cfg.TEST.HAS_RPN:
       blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

def im_attr(net, im, boxes=None):
    """Predict face attributes in an image given part proposals

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        probs (ndarray): 40 x 2 array of face attributes probabilities
    """
    blobs, im_scales = _get_blobs(im, boxes)
    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)

    probs = np.zeros((0, 2), dtype=np.float32)
    probs = np.vstack((probs, blobs_out['5_o_clock_shadow_prob']))
    probs = np.vstack((probs, blobs_out['arched_eyebrows_prob']))
    probs = np.vstack((probs, blobs_out['attractive_prob']))
    probs = np.vstack((probs, blobs_out['bags_under_eyes_prob']))
    probs = np.vstack((probs, blobs_out['bald_prob']))
    probs = np.vstack((probs, blobs_out['bangs_prob']))
    probs = np.vstack((probs, blobs_out['big_lips_prob']))
    probs = np.vstack((probs, blobs_out['big_nose_prob']))
    probs = np.vstack((probs, blobs_out['black_hair_prob']))
    probs = np.vstack((probs, blobs_out['blond_hair_prob']))

    probs = np.vstack((probs, blobs_out['blurry_prob']))
    probs = np.vstack((probs, blobs_out['brown_hair_prob']))
    probs = np.vstack((probs, blobs_out['bushy_eyebrows_prob']))
    probs = np.vstack((probs, blobs_out['chubby_prob']))
    probs = np.vstack((probs, blobs_out['double_chin_prob']))
    probs = np.vstack((probs, blobs_out['eyeglasses_prob']))
    probs = np.vstack((probs, blobs_out['goatee_prob']))
    probs = np.vstack((probs, blobs_out['gray_hair_prob']))
    probs = np.vstack((probs, blobs_out['heavy_makeup_prob']))
    probs = np.vstack((probs, blobs_out['high_cheekbones_prob']))

    probs = np.vstack((probs, blobs_out['male_prob']))
    probs = np.vstack((probs, blobs_out['mouth_slightly_open_prob']))
    probs = np.vstack((probs, blobs_out['mustache_prob']))
    probs = np.vstack((probs, blobs_out['narrow_eyes_prob']))
    probs = np.vstack((probs, blobs_out['no_beard_prob']))
    probs = np.vstack((probs, blobs_out['oval_face_prob']))
    probs = np.vstack((probs, blobs_out['pale_skin_prob']))
    probs = np.vstack((probs, blobs_out['pointly_nose_prob']))
    probs = np.vstack((probs, blobs_out['receding_hairline_prob']))
    probs = np.vstack((probs, blobs_out['rosy_cheeks_prob']))

    probs = np.vstack((probs, blobs_out['sideburns_prob']))
    probs = np.vstack((probs, blobs_out['smiling_prob']))
    probs = np.vstack((probs, blobs_out['straight_hair_prob']))
    probs = np.vstack((probs, blobs_out['wavy_hair_prob']))
    probs = np.vstack((probs, blobs_out['wearing_earrings_prob']))
    probs = np.vstack((probs, blobs_out['wearing_hat_prob']))
    probs = np.vstack((probs, blobs_out['wearing_lipstick_prob']))
    probs = np.vstack((probs, blobs_out['wearing_necklace_prob']))
    probs = np.vstack((probs, blobs_out['wearing_necktie_prob']))
    probs = np.vstack((probs, blobs_out['young_prob']))

    select_rois = net.blobs['select_rois'].data
    #filters = net.params['male_attr'][0].data
    #filters = filters.reshape((512, 256, 6, 6))
    #print filters.shape
    #vis_square(filters.transpose(0,2,3,1)[:,:,:,:3])
    return probs, select_rois

def im_detect(net, im, boxes=None):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)

    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    else:
        # use softmax estimated probabilities
        scores = blobs_out['cls_prob']

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    return scores, pred_boxes

def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            # CPU NMS is much faster than GPU NMS when the number of boxes
            # is relative small (e.g., < 10k)
            # TODO(rbg): autotune NMS dispatch
            keep = nms(dets, thresh, force_cpu=True)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes

def test_net(net, imdb, max_per_image=100, thresh=0.05, vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all attributes are collected into:
    #    all_probs[image] = 40 x 2 array of attributes in
    #    (score1, score2)
    all_probs = [[] for _ in xrange(num_images)]


    output_dir = get_output_dir(imdb, net)

    # timers
    _t = {'im_attr' : Timer()}

    if not cfg.TEST.HAS_RPN:
        roidb = imdb.roidb

    # dataset eval
    # imdb.dataset_eval()

    for i in xrange(1):
        # filter out any ground truth boxes
        if cfg.TEST.HAS_RPN:
            box_proposals = None
        else:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select those the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]


        k = 2001
        im = cv2.imread(imdb.image_path_at(k))
        # resize to 178*218
        # im_resized = cv2.resize(im, (178, 218))

        _t['im_attr'].tic()
        probs, select_rois = im_attr(net, im, box_proposals)
        _t['im_attr'].toc()
        # print probs
        # print select_rois

        all_probs[k] = probs
        # draw rois detected on the image
        '''for j in xrange(select_rois.shape[0]-1):
                source_im = Image.open(imdb.image_path_at(k))
                source_draw = ImageDraw.Draw(source_im)
                source_draw.rectangle(select_rois[j,1:] / 3.37, outline=0x2c2cee)
                source_im.show()
        '''
        # write attributes
        attributes_present = np.zeros(probs.shape[0])
        for j in xrange(probs.shape[0]):
            if probs[j, 1] >= probs[j, 0]:
                attributes_present[j] = 1
        attributes_gt = imdb.load_celeba_annotation(imdb.image_index[k])['face_attrs']
        with open(os.path.join('./results', str(k) + '.txt'), 'w') as file:
            file.write('{}\n'.format(imdb.image_path_at(k)))
            file.write('\\hline\n')
            file.write('{\\bf Attribute} &{\\bf Prediction} & {\\bf Ground-truth} \
                        &{\\bf Attribute} &{\\bf Prediction} &{\\bf Ground-truth} \\\\ \n')
            print '{}'.format(imdb.image_path_at(k))

            for j in xrange(len(imdb.face_attributes_name())):
                if attributes_present[j] == 1:
                    present = 'Yes'
                else:
                    present = 'No'
                if attributes_gt[j] == 1:
                    gt = 'Yes'
                else:
                    gt = 'No'
                if j % 2 == 0:
                    file.write('\\hline\n')
                    file.write('{} & {} & {} &'.format(imdb.face_attributes_name()[j], present, gt))
                else:
                    file.write('{} & {} & {} \\\\ \n'.format(imdb.face_attributes_name()[j], present, gt))

                print '{} {}'.format(imdb.face_attributes_name()[j], present)
            file.write('\\hline')

        print 'im_attr: {:d}/{:d} {:.3f}s' \
              .format(i + 1, num_images, _t['im_attr'].average_time)

    attr_file = os.path.join(output_dir, 'attributes.pkl')
    with open(attr_file, 'wb') as f:
        cPickle.dump(all_probs, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating attributes'
    imdb.evaluate_attributes(all_probs, output_dir)
