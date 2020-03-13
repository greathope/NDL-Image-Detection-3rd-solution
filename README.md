# 3rd place solution for [国立国会図書館の画像データレイアウト認識](https://signate.jp/competitions/218)

## My machine 
  * Intel(R) Core(TM) i7-6850K CPU @ 3.60GHz
  * CUDA  10.1, python3.6.9, pytorch 1.3.1
  * 2 1080ti
  

## Step 0: Set up environment
```
1. install anaconda
2. conda create -n fuxian python=3.6.9
3. conda activate fuxian; cd code
4. pip install -r requirements.txt
5. python setup.py develop
6. put all data under the directory data/, and unzip them
```

## Step 1: Prepare dataset

  ```
   python tools/library/prepare_testdataset_split.py
  ```

## Step 2: Inference 
```
1. bash predict.sh
2. python tools/library/merge_bbox_gudian.py
3. python tools/library/merge_bbox_jindai.py 
4. python tools/library/prepare_submit_split_sort.py
```
* you can find the final json in `data/final_submit.json`
## Step 3: Retrain
```
bash data/download.sh
python tools/library/prepare_dataset_split.py
bash train.sh
```
* because I use coco-pretrained weights, so we have to donwload them 
* the training configs use 2 gpu, if you have only 1, you have to change the learning rate to half manually. when training is finished, you can find weight in `data/retrained/models/`


