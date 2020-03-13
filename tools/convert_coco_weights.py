import torch


def convert_faster(src_path, target_path, num_classes):
    model_coco = torch.load(src_path)

    # weight
    model_coco["state_dict"]["bbox_head.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.fc_cls.weight"][
                                                            :num_classes, :]
    # # bias
    model_coco["state_dict"]["bbox_head.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.fc_cls.bias"][
                                                          :num_classes]

    # save new model
    torch.save(model_coco, target_path)


def convert_cascade(src_path, target_path, num_classes):
    model_coco = torch.load(src_path)

    # weight
    model_coco["state_dict"]["bbox_head.0.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.0.fc_cls.weight"][
                                                            :num_classes, :]
    model_coco["state_dict"]["bbox_head.1.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.1.fc_cls.weight"][
                                                            :num_classes, :]
    model_coco["state_dict"]["bbox_head.2.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.2.fc_cls.weight"][
                                                            :num_classes, :]
    # bias
    model_coco["state_dict"]["bbox_head.0.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.0.fc_cls.bias"][
                                                          :num_classes]
    model_coco["state_dict"]["bbox_head.1.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.1.fc_cls.bias"][
                                                          :num_classes]
    model_coco["state_dict"]["bbox_head.2.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.2.fc_cls.bias"][
                                                          :num_classes]

    # weight
    model_coco["state_dict"]["bbox_head.0.fc_reg.weight"] = model_coco["state_dict"]["bbox_head.0.fc_reg.weight"][
                                                            :num_classes, :]
    model_coco["state_dict"]["bbox_head.1.fc_reg.weight"] = model_coco["state_dict"]["bbox_head.1.fc_reg.weight"][
                                                            :num_classes, :]
    model_coco["state_dict"]["bbox_head.2.fc_reg.weight"] = model_coco["state_dict"]["bbox_head.2.fc_reg.weight"][
                                                            :num_classes, :]
    # bias
    model_coco["state_dict"]["bbox_head.0.fc_reg.bias"] = model_coco["state_dict"]["bbox_head.0.fc_reg.bias"][
                                                          :num_classes]
    model_coco["state_dict"]["bbox_head.1.fc_reg.bias"] = model_coco["state_dict"]["bbox_head.1.fc_reg.bias"][
                                                          :num_classes]
    model_coco["state_dict"]["bbox_head.2.fc_reg.bias"] = model_coco["state_dict"]["bbox_head.2.fc_reg.bias"][
                                                          :num_classes]


    # save new model
    torch.save(model_coco, target_path)

def faster2cascade(src_path, target_path):
    model_coco = torch.load(src_path)
    # print(model_coco.keys())
    unexpected =["bbox_head.fc_cls.weight", "bbox_head.fc_cls.bias", "bbox_head.fc_reg.weight", "bbox_head.fc_reg.bias",
                 "bbox_head.shared_fcs.0.weight", "bbox_head.shared_fcs.0.bias", "bbox_head.shared_fcs.1.weight",
                 "bbox_head.shared_fcs.1.bias"]

    for key in unexpected:
        print(key)
        for i in range(3):
            model_coco['state_dict'][key.replace("bbox_head.", "bbox_head.{}.".format(i))] = model_coco['state_dict'][key]

    for key in unexpected:
        del model_coco['state_dict'][key]

    torch.save(model_coco, target_path)

def neck(src_path, target_path):
    model_coco = torch.load(src_path)
    for key in model_coco['state_dict']:
        if 'neck' in key:
            print(key)


    # torch.save(model_coco, target_path)


if __name__ == "__main__":
    num_classes = 11
    base_dir = '/data/hope/data/data/pretrained/'
    # src_path = base_dir + 'faster_rcnn_dconv_c3-c5_x101_32x4d_fpn_1x_20190201-6d46376f.pth'
    # target_path = base_dir + 'faster_rcnn_dconv_c3-c5_x101_32x4d_{}.pth'.format(num_classes)

    # src_path = base_dir + 'faster_rcnn_x101_64x4d_fpn_1x_20181218-c9c69c8f.pth'
    # target_path = base_dir + 'faster_rcnn_x101_64x4d_{}.pth'.format(num_classes)
    #
    # convert_faster(src_path, target_path, num_classes)

    # src_path = base_dir + 'htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth'
    # target_path = base_dir + 'htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_{}.pth'.format(num_classes)
    # convert_cascade(src_path, target_path, num_classes)


    # src_path = base_dir + 'cascade_rcnn_dconv_c3-c5_r101_fpn_1x_20190125-aaa877cc.pth'
    # target_path = base_dir + 'cascade_rcnn_dconv_c3-c5_r101_fpn_1x_20190125-{}.pth'.format(num_classes)
    # convert_cascade(src_path, target_path, num_classes)

    src_path = base_dir + 'libra_faster_rcnn_r50_fpn_1x_20190610-bf0ea559.pth'
    target_path = base_dir + 'cascade_libra_faster_rcnn_r50_fpn_1x_20190610-bf0ea559.pth'
    faster2cascade(src_path, target_path)
