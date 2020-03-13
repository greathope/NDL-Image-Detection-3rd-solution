# -*- coding: utf-8 -*-
import json
import numpy as np
from bbox_utils import py_cpu_nms as nms
from bbox_utils import box_voting
import os

json_file_list = [
    # '/data/hope/data/data/wine/models/mmdet/01141_x101-32x4d_16e/results_2scales.pkl.bbox.json',
    # '/data/hope/data/data/wine/models/mmdet/01114_x101-32x4d_16e/results_2scales.pkl.bbox.json'
    # '/data/hope/data/data/wine/models/mmdet/01235_cascade_r101_dcn_16e_grid_weight/results_n5_noflip.pkl.bbox.json',
    # '/data/hope/data/data/wine/models/mmdet/01223_reppoints_moment_r101_dcn_fpn_2x/results.pkl.bbox.json'
    '/data/hope/data/data/wine/models/mmdet/01235_cascade_r101_dcn_16e_grid_weight/results_noflip.pkl.bbox.json',
    '/data/hope/data/data/wine/models/mmdet/01252_cascade_r50_dcn_16e_grid_weight/results_noflip.pkl.bbox.json'
]

result_file = '/data/hope/data/data/wine/models/mmdet/ensemble/merge_0127_22.json'

bbox_dict = {}
for index, json_file in enumerate(json_file_list):
    reserve_count = 0
    with open(json_file) as f:
        dataset = json.load(f)
        for x in dataset:
            # if index == 1 and x['score'] < 0.05: continue
            reserve_count += 1
            x['bbox'][2] = x['bbox'][2] + x['bbox'][0] - 1
            x['bbox'][3] = x['bbox'][3] + x['bbox'][1] - 1
            x['bbox'].append(x['score'])
            image_id = x['image_id']
            category_id = x['category_id']

            if bbox_dict.get(image_id) is None:
                bbox_dict[image_id] = {}
            if bbox_dict[image_id].get(category_id) is None:
                bbox_dict[image_id][category_id] = []
            bbox_dict[image_id][category_id].append(x['bbox'])
        print('length of dataset:', json_file, len(dataset), ' resvered box:', reserve_count)

json_results = []
box_count = 0
for key in bbox_dict:
    for label in bbox_dict[key]:
        if len(bbox_dict[key][label]) > 0:
            nms_in = np.array(bbox_dict[key][label], dtype=np.float32, copy=True)
            keep = nms(nms_in, 0.3)
            nms_out = nms_in[keep, :]
            # nms_out=soft_nms(nms_in,0.5,method='linear')
            vote_out = box_voting(nms_out, nms_in, thresh=0.8, scoring_method='ID')
            for box in vote_out:
                w = box[2] - box[0] + 1
                h = box[3] - box[1] + 1
                data = dict()
                data['image_id'] = key
                data['bbox'] = [round(float(box[0]), 4), round(float(box[1]), 4), round(float(w), 4),
                                round(float(h), 4)]
                data['score'] = float(box[4])  # /len(json_file_list)
                data['category_id'] = label
                # data['category_id'] = label+1
                json_results.append(data)
                box_count += 1
print('out box count', box_count)
# print(results[0]['1.jpg'])

print('writing results to {}'.format(result_file))
# mmcv.dump(outputs, args.out)
# results2json(dataset, outputs, result_file)
with open(result_file, 'w') as f:
    # json.dump(json_results, f)
    json_str = json.dumps(json_results)
    f.write(json_str)