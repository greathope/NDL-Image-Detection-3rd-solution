import json
from collections import defaultdict
import os
from PIL import Image, ImageDraw

# exp = '01222_cascade_r101_dcn_16e_grid'
exp = '01231_cascade_r101_dcn_16e_grid_gcb'

base_dir = '/data/hope/data/data/wine/data/'
test = json.load(open(base_dir + 'chongqing1_round1_testA_20191223/testA.json'))
id2im_name = {i['id']:i['file_name'] for i in test['images']}
name2wh = {i['file_name']:[i['width'], i['height']] for i in test['images']}


print("loading bboxes...")
res_bboxes = json.load(open('/data/hope/data/data/wine/models/mmdet/{}/results.pkl.bbox.json'.format(exp)))
# res_bboxes = json.load(open('/data/hope/data/data/wine/models/mmdet/01221_cascade_r101_dcn_16e_cj/results_cj.pkl.bbox.json'))

# res_bboxes = json.load(open('/data/hope/data/data/wine/models/mmdet/01231_cascade_r101_dcn_16e_grid_gcb/results.pkl.bbox.json'))
# res_bboxes = json.load(open('/data/hope/data/data/wine/models/mmdet/01223_reppoints_moment_r101_dcn_fpn_2x/results.pkl.bbox.json'))
im_name2bboxes = defaultdict(list)


for bbox in res_bboxes:
    im_name = id2im_name[bbox['image_id']]
    category_id = bbox['category_id']

    ####
    if bbox['score'] < 0.03: continue
    ###

    if name2wh[im_name][0] > 4000 and category_id not in [6, 7, 8]:
        continue

    if name2wh[im_name][0] < 4000 and category_id in [6, 7, 8]:
        continue
    im_name2bboxes[im_name].append([bbox['score'], category_id] + bbox['bbox'])


images, annotations = [], []

im_dir = base_dir + 'chongqing1_round1_testA_20191223/images/'
save_dir = base_dir + 'chongqing1_round1_testA_20191223/vis_' + exp
os.makedirs(save_dir, exist_ok=True)
for im_name, bboxes in im_name2bboxes.items():
    max_score = max([bbox[0] for bbox in bboxes])
    if max_score < 0.2:
        continue

    im = Image.open(im_dir + im_name)
    draw = ImageDraw.Draw(im)

    for bbox in bboxes:
        category_id = bbox[1]
        xmin, ymin = bbox[2], bbox[3]
        xmax, ymax = xmin + bbox[4], ymin + bbox[5]
        draw.rectangle([xmin, ymin, xmax, ymax], outline='red')
    im.save(os.path.join(save_dir, im_name))






# print(img_counter)
# print("{} bboxes".format(bbox_counter))
#
# predictions = {"images":images, "annotations":annotations}
#
# save_dir = base_dir + 'submit'
# os.makedirs(save_dir, exist_ok=True)
# # save_path = os.path.join(save_dir, 'submit_0123_22.json')
# save_path = os.path.join(save_dir, 'sub.json')
# print("save predictions to {}".format(save_path))
# with open(save_path, 'w') as f:
#     json.dump(predictions, f)


# 2265 38998  01222_cascade_r101_dcn_16e_grid