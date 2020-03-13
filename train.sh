#!/usr/bin/env bash

# gudian
CUDA_VISIBLE_DEVICES=0,2 bash ./tools/dist_train.sh configs/library/gudian/01311_cascade_r50_dcn_gudian.py 2 --seed 2020
CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/library/gudian/01312_cascade_r101_dcn_gudian.py 2 --seed 2020
CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/library/gudian/01313_cascade_r101_dcn_anchor_gudian.py 2 --seed 2020
CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/library/gudian/02021_cascade_x101_anchor_gudian.py.py 2 --seed 2020
CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/library/gudian/02022_cascade_x101_64_anchor_gudian.py 2 --seed 2020
CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/library/gudian/02264_cascade_r101_dcn_gudian.py 2 --seed 2020
CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/library/gudian/02265_cascade_x101_32_gudian.py 2 --seed 2020
CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/library/gudian/02266_cascade_x101_64_gudian.py 2 --seed 2020


CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/library/jindai/01314_cascade_r101_anchor_2x_jindai.py 2 --seed 2020
CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/library/jindai/02011_cascade_r101_dcn_anchor_2x_jindai.py 2 --seed 2020
CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/library/jindai/02012_cascade_x101_64_anchor_2x_jindai.py 2 --seed 2020
CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/library/jindai/02241_cascade_r50_dcn_anchor_2x_jindai.py 2 --seed 2020
CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/library/jindai/02242_cascade_r101_anchor_2x_jindai.py 2 --seed 2020
CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/library/jindai/02243_cascade_r101_dcn_anchor_2x_jindai.py 2 --seed 2020
CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/library/jindai/02244_cascade_x101_32_anchor_2x_jindai.py 2 --seed 2020
CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/library/jindai/02245_cascade_r50_anchor_2x_jindai.py 2 --seed 2020
CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/library/jindai/02246_cascade_x101_64_anchor_2x_jindai.py 2 --seed 2020
CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/library/jindai/013110_cascade_x101_32_anchor_2x_jindai.py 2 --seed 2020


