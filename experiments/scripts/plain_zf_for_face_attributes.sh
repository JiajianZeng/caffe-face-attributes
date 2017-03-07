#!/bin/bash
# Usage:
# ./experiments/scripts/plain_zf_for_face_attributes.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is only celeba, lfwa for now
#
# Example:
# ./experiments/scripts/plain_zf_for_face_attributes.sh 0 ZF celeba \
#   --set EXP_DIR plain_zf RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  celeba)
    TRAIN_IMDB="celeba_trainval"
    TEST_IMDB="celeba_test"
    PT_DIR="celeba"
    ;;
  lfwa)
    TRAIN_IMDB="lfwa_train"
    TEST_IMDB="lfwa_test"
    PT_DIR="lfwa"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/plain_zf_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# for celeba, fine-tuned from PAN, fix BNN
# --weights data/imagenet_models/${NET}.v2.caffemodel \
#time ./tools/train_plain_zf_for_face_attributes.py --gpu ${GPU_ID} \
#  --net_name ${NET} \
#  --weights data/imagenet_models/${NET}.v2.caffemodel \
#  --imdb ${TRAIN_IMDB} \
#  --cfg experiments/cfgs/plain_zf_for_face_attributes.yml \
#  ${EXTRA_ARGS}

# for lfwa
# time ./tools/train_plain_zf_for_face_attributes.py --gpu ${GPU_ID} \
#  --net_name ${NET} \
#  --weights output/plain_zf/celeba_trainval/ZF_plain_zf_final.caffemodel \
#  --imdb ${TRAIN_IMDB} \
#  --cfg experiments/cfgs/plain_zf_for_face_attributes.yml \
#  ${EXTRA_ARGS}

set +x
#NET_FINAL=`grep "Final model:" ${LOG} | awk '{print $3}'`
NET_FINAL="output/plain_zf/celeba_trainval/plain_zf_plain_zf_iter_100000.caffemodel"
set -x

time ./tools/test_net_plain_zf.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/plain_zf/plain_zf_for_face_attributes_test.pt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/plain_zf_for_face_attributes.yml \
  ${EXTRA_ARGS}
