import json
from collections import defaultdict
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

base_dir = '/data/hope/data/data/library/'
# test = json.load(open(base_dir + 'test.json'))
test = json.load(open('/data/hope/test.json'))
id2im_name = {i['id']:i['file_name'] for i in test['images']}
# name2wh = {i['file_name']:[i['width'], i['height']] for i in test['images']}
print(len(id2im_name.values()))

print("loading bboxes...")
# res_bboxes = json.load(open('/data/hope/data/data/library/models/01241_cascade_r50_dcn_16e/results.pkl.bbox.json'))
res_bboxes = json.load(open('/data/hope/results.pkl.bbox.json'))

im_name2bboxes = defaultdict(list)
for bbox in res_bboxes:
    im_name = id2im_name[bbox['image_id']]
    if bbox['score'] < 0.01:
        continue
    im_name2bboxes[im_name].append(bbox['bbox'] + [bbox['category_id']])

font = ImageFont.truetype('/data/hope/Verdana.ttf',24)

save_dir = base_dir + 'vis_test/'
os.makedirs(save_dir, exist_ok=True)
os.system('rm ' + save_dir + '*')
img_counter = 0
for im_name in id2im_name.values():
    bboxes = im_name2bboxes[im_name]

    # img_counter += 1
    im = Image.open(base_dir + 'test_images/' + im_name).convert('RGB')
    draw = ImageDraw.Draw(im)

    for bbox in bboxes:
        category_id = bbox[-1]
        xmin, ymin = bbox[0], bbox[1]
        xmax, ymax = xmin + bbox[2], ymin + bbox[3]
        draw.rectangle([xmin, ymin, xmax, ymax], outline='green')
        draw.text((xmin, ymin), str(category_id), (255, 0, 0), font=font)

    im.save(save_dir + im_name)
