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

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
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
    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
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

    return probs

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

    for i in xrange(num_images):
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

        im = cv2.imread(imdb.image_path_at(i))
        # resize to 178*218
        # im_resized = cv2.resize(im, (178, 218))

        _t['im_attr'].tic()
        probs = im_attr(net, im, box_proposals)
        _t['im_attr'].toc()

        all_probs[i] = probs

        print 'im_attr: {:d}/{:d} {:.3f}s' \
              .format(i + 1, num_images, _t['im_attr'].average_time)

    attr_file = os.path.join(output_dir, 'attributes.pkl')
    with open(attr_file, 'wb') as f:
        cPickle.dump(all_probs, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating attributes'
    imdb.evaluate_attributes(all_probs, output_dir)
