import cv2
from glob import glob
from tqdm import tqdm
import json
import os
from collections import defaultdict


def convert_to_coco(name2bboxes, base_dir, labels_mp,  name):
    idx = 1
    image_id = 20200000000
    images = []
    annotations = []
    for im_path, bboxes in tqdm(name2bboxes.items()):
        if len(bboxes) == 0:
            print(im_path)
            continue

        im_name = im_path.split('/')[-1]
        im = cv2.imread(im_path)
        h, w, _ = im.shape
        image_id += 1
        image = {'file_name': im_name, 'width': w, 'height': h, 'id': image_id}
        images.append(image)


        for bbox in bboxes:
            x1, y1, x2, y2 = bbox[:4]
            category = bbox[-1]

            bbox = [x1, y1, x2 - x1, y2 - y1]
            seg = []
            # bbox[] is x,y,w,h
            # left_top
            seg.append(bbox[0])
            seg.append(bbox[1])
            # left_bottom
            seg.append(bbox[0])
            seg.append(bbox[1] + bbox[3])
            # right_bottom
            seg.append(bbox[0] + bbox[2])
            seg.append(bbox[1] + bbox[3])
            # right_top
            seg.append(bbox[0] + bbox[2])
            seg.append(bbox[1])
            ann = {'segmentation': [seg], 'area': bbox[2] * bbox[3], 'iscrowd': 0, 'image_id': image_id,
                   'bbox': bbox, 'category_id': labels_mp[category], 'id': idx, 'ignore': 0}
            idx += 1
            annotations.append(ann)
    # print(max(max_gt))
    ann = dict()
    ann['type'] = 'instances'
    ann['images'] = images
    ann['annotations'] = annotations
    category = [{'supercategory': 'none', 'id': i, 'name': label} for label, i in labels_mp.items()]
    ann['categories'] = category
    json.dump(ann, open(os.path.join(base_dir, 'train_{}.json'.format(name)), 'w'))


base_dir = 'data/'
gt = glob(base_dir + 'train_annotations/*.json')

gudian2bboxes = dict()   # 古典
jindai2bboxes = dict()   # 近代

gudian_labels = ['1_overall', '2_handwritten', '3_typography', '4_illustration', '5_stamp']
gudian_labels_mp = {label: i for i, label in enumerate(gudian_labels, 1)}

jindai_labels = ['1_overall', '4_illustration', '5_stamp', '6_headline', '7_caption', '8_textline', '9_table']
jindai_labels_mp = {label: i for i, label in enumerate(jindai_labels, 1)}

gudian_counter = 0
jindai_counter = 0
for gt_path in gt:
    im_path = gt_path.replace('train_annotations', 'train_images').replace('json', 'jpg')
    infos = json.load(open(gt_path))
    time = infos['attributes']["年代"]

    bboxes = []
    labels = infos['labels']
    for label in labels:
        box2d = label['box2d']
        x1, y1, x2, y2 = box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']
        category = label['category']
        bboxes.append([x1, y1, x2, y2, category])

    if time == '古典籍':
        gudian2bboxes[im_path] = bboxes
        gudian_counter += 1
    elif time == '近代':
        jindai2bboxes[im_path] = bboxes
        jindai_counter += 1

print(gudian_counter, jindai_counter)

convert_to_coco(gudian2bboxes, base_dir, gudian_labels_mp, 'gudian')
convert_to_coco(jindai2bboxes, base_dir, jindai_labels_mp, 'jindai')