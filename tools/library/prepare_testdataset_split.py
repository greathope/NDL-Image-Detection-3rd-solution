from glob import glob
from tqdm import tqdm
import cv2
import json
import os
from PIL import Image


def make_coco_testdataset(img_paths, base_dir, labels_mp, name):
    idx = 1
    image_id = 20200000000
    images = []
    annotations = []
    for img_path in tqdm(img_paths):
        im = cv2.imread(img_path)
        h, w, _ = im.shape

        image_id += 1
        image = {'file_name': os.path.basename(img_path), 'width': w, 'height': h, 'id': image_id}
        images.append(image)

        anno_ = {'segmentation': [[]], 'area': 100, 'iscrowd': 0, 'image_id': image_id,
               'bbox': [10, 10, 10, 10], 'category_id': 1, 'id': idx, 'ignore': 0}
        idx += 1
        annotations.append(anno_)

    ann = dict()
    ann['type'] = 'instances'
    ann['images'] = images
    ann['annotations'] = annotations
    category = [{'supercategory': 'none', 'id': i, 'name': label} for label, i in labels_mp.items()]
    ann['categories'] = category
    json.dump(ann, open(base_dir + '{}.json'.format(name),'w'))


base_dir = 'data/'
gt = glob(base_dir + 'test_annotations/*.json')

gudian_labels = ['1_overall', '2_handwritten', '3_typography', '4_illustration', '5_stamp']
gudian_labels_mp = {label: i for i, label in enumerate(gudian_labels, 1)}

jindai_labels = ['1_overall', '4_illustration', '5_stamp', '6_headline', '7_caption', '8_textline', '9_table']
jindai_labels_mp = {label: i for i, label in enumerate(jindai_labels, 1)}

gudian_counter, jindai_counter = 0, 0
gudian_imgs, jindai_imgs = [], []
for gt_path in gt:
    im_name = gt_path.replace('test_annotations', 'test_images').replace('json', 'jpg')
    infos = json.load(open(gt_path))
    time = infos['attributes']["年代"]

    if time == '古典籍':
        gudian_imgs.append(im_name)
        gudian_counter += 1
    elif time == '近代':
        jindai_imgs.append(im_name)
        jindai_counter += 1

print(gudian_counter, jindai_counter)

make_coco_testdataset(gudian_imgs, base_dir, gudian_labels_mp, 'test_gudian')
make_coco_testdataset(jindai_imgs, base_dir, jindai_labels_mp, 'test_jindai')