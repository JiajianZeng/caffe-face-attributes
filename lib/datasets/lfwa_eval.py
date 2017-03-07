# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os
import cPickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, average_precision_score

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    attributes = []
    for attr in tree.findall('attribute'):
        attr_struct = {}
        attr_struct['name'] = attr.find('name').text
        attr_struct['value'] = int(attr.find('value').text)
        attributes.append(attr_struct)

    return attributes

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(attrpath,
             annopath,
             imagesetfile,
             attr_name,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots_lfwa.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[i] = parse_rec(annopath.format(imagename[0:-4]))
            if i % 100 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames))
        # save
        print 'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            recs = cPickle.load(f)

    # extract gt value for this attribute
    attr_recs = {}
    npos = 0
    for i, imagename in enumerate(imagenames):
        R = [attr for attr in recs[i] if attr['name'] == attr_name]
        attr = np.array([x['value'] for x in R])
        npos = npos + np.sum(attr)
        attr_recs[i] = {'attr': attr}

    # read attrs
    attrfile = attrpath.format(attr_name)
    with open(attrfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_names = [x[0] for x in splitlines]
    attr_probs = np.array([[float(z) for z in x[1:]] for x in splitlines])

    # go down attrs and mark TPs and FPs
    nd = len(image_names)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    tn = np.zeros(nd)
    fn = np.zeros(nd)
    for d in range(nd):
        R = attr_recs[d]
        attr_prob = attr_probs[d, :].astype(float)
        ATTRGT = R['attr'].astype(float)

        if attr_prob[0] > attr_prob[1]:
            ATTRPRE = 0
        else:
            ATTRPRE = 1

        if ATTRGT == 1 and ATTRPRE == 1:
            tp[d] = 1
        if ATTRGT == 0 and ATTRPRE == 1:
            fp[d] = 1
        if ATTRGT == 1 and ATTRPRE == 0:
            fn[d] = 1
        if ATTRGT == 0 and ATTRPRE == 0:
            tn[d] = 1


    # compute precision recall
    # fp = np.cumsum(fp)
    # tp = np.cumsum(tp)
    fp = np.sum(fp)
    tp = np.sum(tp)
    tn = np.sum(tn)
    fn = np.sum(fn)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    # ap = voc_ap(rec, prec, use_07_metric)
    acc = (tp + tn) / float(nd)

    return rec, prec, acc

def _plot_roc_curve_single_class(y_gt, y_score, label_name):
    fpr, tpr, _ = roc_curve(y_gt, y_score)
    roc_auc = auc(fpr, tpr)
    # plot roc curve
    plt.figure()
    plt.plot(fpr, tpr, label='{:s} (auc area = {:.2f})'.format(label_name, roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for {:s}'.format(label_name))
    plt.legend(loc='lower right')
    plt.show()

def _plot_roc_curve_multi_class(y_gt, y_score, label_name):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure()
    for i in range(y_gt.shape[0]):
        fpr[i], tpr[i], _ = roc_curve(y_gt[i, :], y_score[i, :])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label='{:s} (auc area = {:.2f})'.format(label_name[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi Class ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

def roc_curve_single_attribute(attrpath,
                               annopath,
                               imagesetfile,
                               attr_name,
                               cachedir):
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots_lfwa.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[i] = parse_rec(annopath.format(imagename[0:-4]))
            if i % 100 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames))
        # save
        print 'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            recs = cPickle.load(f)

    # extract gt value for this attribute
    attr_recs = {}
    for i, imagename in enumerate(imagenames):
        R = [attr for attr in recs[i] if attr['name'] == attr_name]
        attr = np.array([x['value'] for x in R])
        attr_recs[i] = {'attr': attr}

    y_gt = np.array([attr_recs[i]['attr'][0] for i, imagename in enumerate(imagenames)])

    # read attrs
    attrfile = attrpath.format(attr_name)
    with open(attrfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    y_score = np.array([float(x[2]) for x in splitlines])

    _plot_roc_curve_single_class(y_gt, y_score, attr_name)
    return average_precision_score(y_gt, y_score)

def roc_curve_multi_attribute(attrpath,
                               annopath,
                               imagesetfile,
                               attr_name,
                               cachedir):
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots_lfwa.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[i] = parse_rec(annopath.format(imagename[0:-4]))
            if i % 100 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames))
        # save
        print 'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            recs = cPickle.load(f)

    # extract gt value for this attribute
    n_attr = attr_name.shape[0]
    n_sample = len(imagenames)
    y_gt = np.zeros([n_attr, n_sample], dtype=np.int32)
    y_score = np.zeros([n_attr, n_sample], dtype=np.float32)


    for i in range(n_attr):
        attr_recs = {}
        for i, imagename in enumerate(imagenames):
            R = [attr for attr in recs[i] if attr['name'] == attr_name[i]]
            attr = np.array([x['value'] for x in R])
            attr_recs[i] = {'attr': attr}
        for j, imagename in enumerate(imagenames):
            y_gt[i, j] = attr_recs[i]['attr'][0]

        # read attrs
        attrfile = attrpath.format(attr_name[i])
        with open(attrfile, 'r') as f:
            lines = f.readlines()

        splitlines = [x.strip().split(' ') for x in lines]
        for j, x in enumerate(splitlines):
            y_score[i, j] = float(x[2])

    _plot_roc_curve_multi_class(y_gt, y_score, attr_name)

def dataset_eval(annopath,
                 imagesetfile,
                 attr_name,
                 cachedir):

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots_lfwa_trainval.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[i] = parse_rec(annopath.format(imagename[0:-4]))
            if i % 100 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames))
        # save
        print 'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            recs = cPickle.load(f)

    # extract gt value for this attribute
    npos = 0
    for i, imagename in enumerate(imagenames):
        R = [attr for attr in recs[i] if attr['name'] == attr_name]
        attr = np.array([x['value'] for x in R])
        npos = npos + np.sum(attr)

    # go down attrs and mark TPs and FPs
    nd = len(imagenames)
    ratio_positive = npos / float(nd)

    return nd, ratio_positive