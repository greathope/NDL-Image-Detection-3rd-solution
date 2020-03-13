#!/usr/bin/env bash


# gudian
CUDA_VISIBLE_DEVICES=0 bash ./tools/dist_test.sh configs/library/gudian/01311_cascade_r50_dcn_gudian.py \
   data/fuxian/models/gudian/1311_no.pth 1 \
    --out data/fuxian/output/gudian/1311.pkl --eval bbox

CUDA_VISIBLE_DEVICES=0 bash ./tools/dist_test.sh configs/library/gudian/01312_cascade_r101_dcn_gudian.py \
   data/fuxian/models/gudian/1312_no.pth 1 \
    --out data/fuxian/output/gudian/1312.pkl --eval bbox

CUDA_VISIBLE_DEVICES=0 bash ./tools/dist_test.sh configs/library/gudian/01313_cascade_r101_dcn_anchor_gudian.py \
   data/fuxian/models/gudian/1313_no.pth 1 \
    --out data/fuxian/output/gudian/1313.pkl --eval bbox

CUDA_VISIBLE_DEVICES=0 bash ./tools/dist_test.sh configs/library/gudian/02021_cascade_x101_anchor_gudian.py \
  data/fuxian/models/gudian/2021_no.pth 1 \
    --out data/fuxian/output/gudian/2021.pkl --eval bbox

CUDA_VISIBLE_DEVICES=0 bash ./tools/dist_test.sh configs/library/gudian/02022_cascade_x101_64_anchor_gudian.py \
   data/fuxian/models/gudian/2022_no.pth 1 \
    --out data/fuxian/output/gudian/2022.pkl --eval bbox

CUDA_VISIBLE_DEVICES=0 bash ./tools/dist_test.sh configs/library/gudian/02264_cascade_r101_dcn_gudian.py \
   data/fuxian/models/gudian/2264_no.pth 1 \
    --out data/fuxian/output/gudian/2264.pkl --eval bbox

CUDA_VISIBLE_DEVICES=0 bash ./tools/dist_test.sh configs/library/gudian/02265_cascade_x101_32_gudian.py \
   data/fuxian/models/gudian/2265_no.pth 1 \
    --out data/fuxian/output/gudian/2265.pkl --eval bbox

CUDA_VISIBLE_DEVICES=0 bash ./tools/dist_test.sh configs/library/gudian/02266_cascade_x101_64_gudian.py \
   data/fuxian/models/gudian/2266_no.pth 1 \
    --out data/fuxian/output/gudian/2266.pkl --eval bbox


# jindai

CUDA_VISIBLE_DEVICES=0 bash ./tools/dist_test.sh configs/library/jindai/01314_cascade_r101_anchor_2x_jindai.py \
   data/fuxian/models/jindai/1314_no.pth 1 \
    --out data/fuxian/output/jindai/1314.pkl --eval bbox

CUDA_VISIBLE_DEVICES=0 bash ./tools/dist_test.sh configs/library/jindai/02011_cascade_r101_dcn_anchor_2x_jindai.py \
   data/fuxian/models/jindai/2011_no.pth 1 \
    --out data/fuxian/output/jindai/2011.pkl --eval bbox

CUDA_VISIBLE_DEVICES=0 bash ./tools/dist_test.sh configs/library/jindai/02012_cascade_x101_64_anchor_2x_jindai.py \
   data/fuxian/models/jindai/2012_no.pth 1 \
    --out data/fuxian/output/jindai/2012.pkl --eval bbox

CUDA_VISIBLE_DEVICES=0 bash ./tools/dist_test.sh configs/library/jindai/02241_cascade_r50_dcn_anchor_2x_jindai.py \
   data/fuxian/models/jindai/2241_no.pth 1 \
    --out data/fuxian/output/jindai/2241.pkl --eval bbox

CUDA_VISIBLE_DEVICES=0 bash ./tools/dist_test.sh configs/library/jindai/02242_cascade_r101_anchor_2x_jindai.py \
   data/fuxian/models/jindai/2242_no.pth 1 \
    --out data/fuxian/output/jindai/2242.pkl --eval bbox

CUDA_VISIBLE_DEVICES=0 bash ./tools/dist_test.sh configs/library/jindai/02243_cascade_r101_dcn_anchor_2x_jindai.py \
   data/fuxian/models/jindai/2243_no.pth 1 \
    --out data/fuxian/output/jindai/2243.pkl --eval bbox

CUDA_VISIBLE_DEVICES=0 bash ./tools/dist_test.sh configs/library/jindai/02244_cascade_x101_32_anchor_2x_jindai.py \
   data/fuxian/models/jindai/2244_no.pth 1 \
    --out data/fuxian/output/jindai/2244.pkl --eval bbox

CUDA_VISIBLE_DEVICES=0 bash ./tools/dist_test.sh configs/library/jindai/02245_cascade_r50_anchor_2x_jindai.py \
   data/fuxian/models/jindai/2245_no.pth 1 \
    --out data/fuxian/output/jindai/2245.pkl --eval bbox

CUDA_VISIBLE_DEVICES=0 bash ./tools/dist_test.sh configs/library/jindai/02246_cascade_x101_64_anchor_2x_jindai.py \
   data/fuxian/models/jindai/2246_no.pth 1 \
    --out data/fuxian/output/jindai/2246.pkl --eval bbox

CUDA_VISIBLE_DEVICES=0 bash ./tools/dist_test.sh configs/library/jindai/013110_cascade_x101_32_anchor_2x_jindai.py \
  data/fuxian/models/jindai/13110_no.pth 1 \
    --out data/fuxian/output/jindai/13110.pkl --eval bbox
