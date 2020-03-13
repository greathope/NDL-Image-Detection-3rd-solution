import json
from collections import defaultdict
import os


base_dir = 'data/'
gudian_test = json.load(open(base_dir + 'test_gudian.json'))
gudian_id2im_name = {i['id']:i['file_name'] for i in gudian_test['images']}

gudian_labels = ['1_overall', '2_handwritten', '3_typography', '4_illustration', '5_stamp']
gudian_labels_mp = {i: label for i, label in enumerate(gudian_labels, 1)}

print("loading bboxes...")
res_bboxes = json.load(open('data/merge_all_gudian.json'))
im_name2bboxes = defaultdict(dict)
thr = 0.01
for bbox in res_bboxes:
    im_name = gudian_id2im_name[bbox['image_id']]
    if bbox['score'] < thr: continue
    category_name = gudian_labels_mp[bbox['category_id']]

    if category_name not in im_name2bboxes[im_name]:
        im_name2bboxes[im_name][category_name] = []
    im_name2bboxes[im_name][category_name].append([bbox['bbox'][0], bbox['bbox'][1],
                                                   bbox['bbox'][0] + bbox['bbox'][2],
                                                   bbox['bbox'][1] + bbox['bbox'][3], bbox['score']])

jindai_test = json.load(open(base_dir + 'test_jindai.json'))
jindai_id2im_name = {i['id']:i['file_name'] for i in jindai_test['images']}
jindai_labels = ['1_overall', '4_illustration', '5_stamp', '6_headline', '7_caption', '8_textline', '9_table']
jindai_labels_mp = {i: label for i, label in enumerate(jindai_labels, 1)}

print("loading bboxes...")
res_bboxes = json.load(open('data/merge_all_jindai.json'))
for bbox in res_bboxes:
    im_name = jindai_id2im_name[bbox['image_id']]
    if bbox['score'] < thr: continue
    category_name = jindai_labels_mp[bbox['category_id']]
    if category_name == '9_table': continue

    if category_name not in im_name2bboxes[im_name]:
        im_name2bboxes[im_name][category_name] = []
    im_name2bboxes[im_name][category_name].append([bbox['bbox'][0], bbox['bbox'][1],
                                                   bbox['bbox'][0] + bbox['bbox'][2],
                                                   bbox['bbox'][1] + bbox['bbox'][3], bbox['score']])

print(len(im_name2bboxes))


results = {}

cnt = 0
for name, entry in im_name2bboxes.items():
    temp = {}
    cnt += 1
    for category_name, info in entry.items():
        info = sorted(info, key=lambda x:x[-1], reverse=True)
        info = [i[:-1] for i in info]
        temp[category_name] = info

    results[name] = temp

save_path = os.path.join(base_dir, 'final_submit.json')
print("save predictions to {}".format(save_path))
with open(save_path, 'w') as f:
    json.dump(results, f)
