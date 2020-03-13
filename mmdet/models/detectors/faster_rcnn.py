from ..registry import DETECTORS
from .two_stage import TwoStageDetector
from mmdet.core import (bbox2result, bbox2roi)


@DETECTORS.register_module
class FasterRCNN(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(FasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    def simple_test_bboxes_custom(self, x,
                           img_meta,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False,
                           nms=True):

        rois = bbox2roi(proposals)
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        cls_score, bbox_pred = self.bbox_head(roi_feats)

        if not nms:
            return rois, cls_score, bbox_pred

        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels

    def forward_backbone_(self, img):
        x = self.extract_feat(img)
        return x

    def nms_bboxes(self, img_meta, rois, cls_score, bbox_pred, rcnn_test_cfg):
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=True,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels

    def forward_rpn_(self, x, img_meta):
        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn)
        return proposal_list

    def forward_roi_(self, x, img_meta, proposal_list):
        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=True)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        return bbox_results

    def forward_bbox_(self, x, img_meta, proposal_list):
        rois, cls, bbox = self.simple_test_bboxes_custom(
                x, img_meta, proposal_list, self.test_cfg.rcnn, nms=False)
        return rois, cls, bbox

    def forward_nms_(self, img_meta, rois, cls, bbox):
        det_bboxes, det_labels = self.nms_bboxes(img_meta, rois, cls, bbox, self.test_cfg.rcnn)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        return bbox_results
