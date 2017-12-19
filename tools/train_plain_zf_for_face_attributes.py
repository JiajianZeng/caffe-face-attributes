#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a plain ZF network for face attributes prediction using CelebA or LFWA dataset.
"""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import argparse
import pprint
import numpy as np
import sys, os
import multiprocessing as mp
import cPickle
import shutil
import os.path as osp

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a plain ZF network for face attributes')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--net_name', dest='net_name',
                        help='network name (e.g., "ZF")',
                        default=None, type=str)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def get_roidb(imdb_name):
    imdb = get_imdb(imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    # set proposal method 'gt' to make this imdb load annotation
    imdb.set_proposal_method('gt')

    roidb = get_training_roidb(imdb)
    return roidb, imdb

def get_solvers(net_name):
    n = 'plain_zf'
    # Solvers
    solvers = [[net_name, n, 'plain_zf_for_face_attributes_solver.pt']]
    solvers = [os.path.join(cfg.MODELS_DIR, *s) for s in solvers]
    # Iterations
    max_iters = [100000]
    return solvers, max_iters

# ------------------------------------------------------------------------------
# Pycaffe doesn't reliably free GPU memory when instantiated nets are discarded
# (e.g. "del net" in Python code). To work around this issue, each training
# stage is executed in a separate process using multiprocessing.Process.
# ------------------------------------------------------------------------------

def _init_caffe(gpu_id):
    """Initialize pycaffe in a training process.
    """

    import caffe
    # fix the random seeds (numpy and caffe) for reproducibility
    np.random.seed(cfg.RNG_SEED)
    caffe.set_random_seed(cfg.RNG_SEED)
    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)

def train_plain_zf(queue=None, imdb_name=None, init_model=None, solver=None,
                    max_iters=None, cfg=None):
    """Train a plain ZF for face attributes prediction.
    """
    print 'Init model: {}'.format(init_model)
    print('Using config:')
    pprint.pprint(cfg)

    # initialize caffe
    _init_caffe(cfg.GPU_ID)

    roidb, imdb = get_roidb(imdb_name)
    output_dir = get_output_dir(imdb)
    print 'Output will be saved to `{:s}`'.format(output_dir)
    # Train plain ZF
    model_paths = train_net(solver, roidb, output_dir,
                            pretrained_model=init_model,
                            max_iters=max_iters)
    # Cleanup all but the final model
    #for i in model_paths[:-1]:
        #os.remove(i)
    plain_zf_model_path = model_paths[-1]
    # Send plain ZF model path over the multiprocessing queue
    queue.put({'model_path': plain_zf_model_path})

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    cfg.GPU_ID = args.gpu_id

    # --------------------------------------------------------------------------
    # Pycaffe doesn't reliably free GPU memory when instantiated nets are
    # discarded (e.g. "del net" in Python code). To work around this issue, each
    # training stage is executed in a separate process using
    # multiprocessing.Process.
    # --------------------------------------------------------------------------

    cfg.MODELS_DIR = osp.abspath(osp.join(cfg.ROOT_DIR, 'models', 'celeba'))
    #cfg.MODELS_DIR = osp.abspath(osp.join(cfg.ROOT_DIR, 'models', 'lfwa'))

    # queue for communicated results between processes
    mp_queue = mp.Queue()
    # solves, iters, etc.
    solvers, max_iters = get_solvers(args.net_name)

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Plain ZF, for face attributes prediction using CelebA dataset'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    mp_kwargs = dict(
            queue=mp_queue,
            imdb_name=args.imdb_name,
            init_model=args.pretrained_model,
            solver=solvers[0],
            max_iters=max_iters[0],
            cfg=cfg)
    p = mp.Process(target=train_plain_zf, kwargs=mp_kwargs)
    p.start()
    plain_zf_out = mp_queue.get()
    p.join()

    # Create final model (just a copy of the last stage)
    final_path = os.path.join(
            os.path.dirname(plain_zf_out['model_path']),
            'celeba_plain_zf_final.caffemodel')
    #final_path = os.path.join(
        #os.path.dirname(plain_zf_out['model_path']),
        #'lfwa_plain_zf_final.caffemodel')
    print 'cp {} -> {}'.format(
        plain_zf_out['model_path'], final_path)
    shutil.copy(plain_zf_out['model_path'], final_path)
    print 'Final model: {}'.format(final_path)
