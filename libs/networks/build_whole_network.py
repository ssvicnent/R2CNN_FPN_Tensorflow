# -*-coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from libs.networks import resnet
from libs.networks import mobilenet_v2
from libs.box_utils import encode_and_decode
from libs.box_utils import boxes_utils
from libs.box_utils import anchor_utils
from libs.configs import cfgs
from libs.losses import losses
from libs.box_utils import show_box_in_tensor
from libs.detection_oprations.proposal_opr import postprocess_rpn_proposals
from libs.detection_oprations.anchor_target_layer_without_boxweight import anchor_target_layer
from libs.detection_oprations.proposal_target_layer import proposal_target_layer
from libs.box_utils import nms_rotate


class DetectionNetwork(object):

    def __init__(self, base_network_name, is_training):

        self.base_network_name = base_network_name
        self.is_training = is_training
        self.base_anchor_size_list = cfgs.BASE_ANCHOR_SIZE_LIST
        self.anchor_scales = cfgs.ANCHOR_SCALES
        self.anchor_ratios = cfgs.ANCHOR_RATIOS
        self.stride = cfgs.ANCHOR_STRIDE
        self.level = cfgs.LEVEL
        self.i = 2
        if self.i == len(self.level):
            self.num_anchors_per_location = [len(cfgs.ANCHOR_SCALES[i-2]) * len(cfgs.ANCHOR_RATIOS[i-2]),
                                             len(cfgs.ANCHOR_SCALES[i-1]) * len(cfgs.ANCHOR_RATIOS[i-1])] # len(cfgs.ANCHOR_SCALES[1]) * len(cfgs.ANCHOR_RATIOS[1])
        self.usedropout = cfgs.USE_DROPOUT
        self.img_num = []
    def build_base_network(self, input_img_batch):

        if self.base_network_name.startswith('resnet_v1'):
            return resnet.resnet_base(input_img_batch, scope_name=self.base_network_name, is_training=self.is_training)

        elif self.base_network_name.startswith('MobilenetV2'):
            return mobilenet_v2.mobilenetv2_base(input_img_batch, is_training=self.is_training)

        else:
            raise ValueError('Sry, we only support resnet or mobilenet_v2')

    def postprocess_fastrcnn_h(self, rois, bbox_ppred, scores, img_shape):

        '''

        :param rois:[-1, 4]
        :param bbox_ppred: [-1, (cfgs.Class_num+1) * 4]
        :param scores: [-1, cfgs.Class_num + 1]
        :return:
        '''
        with tf.name_scope('postprocess_fastrcnn_h'):
            rois = tf.stop_gradient(rois)
            scores = tf.stop_gradient(scores)
            bbox_ppred = tf.reshape(bbox_ppred, [-1, cfgs.CLASS_NUM + 1, 4])
            bbox_ppred = tf.stop_gradient(bbox_ppred)

            bbox_pred_list = tf.unstack(bbox_ppred, axis=1)
            score_list = tf.unstack(scores, axis=1)

            allclasses_boxes = []
            allclasses_scores = []
            categories = []
            for i in range(1, cfgs.CLASS_NUM+1):

                # 1. decode boxes in each class
                tmp_encoded_box = bbox_pred_list[i]
                tmp_score = score_list[i]
                tmp_decoded_boxes = encode_and_decode.decode_boxes(encode_boxes=tmp_encoded_box,
                                                                   reference_boxes=rois,
                                                                   scale_factors=cfgs.ROI_SCALE_FACTORS)
                # tmp_decoded_boxes = encode_and_decode.decode_boxes(boxes=rois,
                #                                                    deltas=tmp_encoded_box,
                #                                                    scale_factor=cfgs.ROI_SCALE_FACTORS)

                # 2. clip to img boundaries
                tmp_decoded_boxes = boxes_utils.clip_boxes_to_img_boundaries(decode_boxes=tmp_decoded_boxes,
                                                                             img_shape=img_shape)

                # 3. NMS
                keep = tf.image.non_max_suppression(
                    boxes=tmp_decoded_boxes,
                    scores=tmp_score,
                    max_output_size=cfgs.FAST_RCNN_NMS_MAX_BOXES_PER_CLASS,
                    iou_threshold=cfgs.FAST_RCNN_NMS_IOU_THRESHOLD)

                perclass_boxes = tf.gather(tmp_decoded_boxes, keep)
                perclass_scores = tf.gather(tmp_score, keep)

                allclasses_boxes.append(perclass_boxes)
                allclasses_scores.append(perclass_scores)
                categories.append(tf.ones_like(perclass_scores) * i)

            final_boxes = tf.concat(allclasses_boxes, axis=0)
            final_scores = tf.concat(allclasses_scores, axis=0)
            final_category = tf.concat(categories, axis=0)

            # if self.is_training:
            '''
            in training. We should show the detecitons in the tensorboard. So we add this.
            '''
            kept_indices = tf.reshape(tf.where(tf.greater_equal(final_scores, cfgs.SHOW_SCORE_THRSHOLD)), [-1])
            final_boxes = tf.gather(final_boxes, kept_indices)
            final_scores = tf.gather(final_scores, kept_indices)
            final_category = tf.gather(final_category, kept_indices)

        return final_boxes, final_scores, final_category

    def postprocess_fastrcnn_r(self, rois, bbox_ppred, scores, img_shape):
        '''

        :param rois:[-1, 4]
        :param bbox_ppred: [-1, (cfgs.Class_num+1) * 5]
        :param scores: [-1, cfgs.Class_num + 1]
        :return:
        '''

        with tf.name_scope('postprocess_fastrcnn_r'):
            rois = tf.stop_gradient(rois)
            scores = tf.stop_gradient(scores)
            bbox_ppred = tf.reshape(bbox_ppred, [-1, cfgs.CLASS_NUM + 1, 5])
            bbox_ppred = tf.stop_gradient(bbox_ppred)

            bbox_pred_list = tf.unstack(bbox_ppred, axis=1)
            score_list = tf.unstack(scores, axis=1)

            allclasses_boxes = []
            allclasses_scores = []
            categories = []
            for i in range(1, cfgs.CLASS_NUM+1):

                # 1. decode boxes in each class
                tmp_encoded_box = bbox_pred_list[i]
                tmp_score = score_list[i]
                tmp_decoded_boxes = encode_and_decode.decode_boxes_rotate(encode_boxes=tmp_encoded_box,
                                                                          reference_boxes=rois,
                                                                          scale_factors=cfgs.ROI_SCALE_FACTORS)
                # tmp_decoded_boxes = encode_and_decode.decode_boxes(boxes=rois,
                #                                                    deltas=tmp_encoded_box,
                #                                                    scale_factor=cfgs.ROI_SCALE_FACTORS)

                # 2. clip to img boundaries
                # tmp_decoded_boxes = boxes_utils.clip_boxes_to_img_boundaries(decode_boxes=tmp_decoded_boxes,
                #                                                              img_shape=img_shape)

                # 3. NMS
                keep = nms_rotate.nms_rotate(decode_boxes=tmp_decoded_boxes,
                                             scores=tmp_score,
                                             iou_threshold=cfgs.FAST_RCNN_NMS_IOU_THRESHOLD,
                                             max_output_size=cfgs.FAST_RCNN_NMS_MAX_BOXES_PER_CLASS,
                                             use_angle_condition=False,
                                             angle_threshold=15,
                                             use_gpu=cfgs.ROTATE_NMS_USE_GPU)

                perclass_boxes = tf.gather(tmp_decoded_boxes, keep)
                perclass_scores = tf.gather(tmp_score, keep)

                allclasses_boxes.append(perclass_boxes)
                allclasses_scores.append(perclass_scores)
                categories.append(tf.ones_like(perclass_scores) * i)

            final_boxes = tf.concat(allclasses_boxes, axis=0)
            final_scores = tf.concat(allclasses_scores, axis=0)
            final_category = tf.concat(categories, axis=0)

            # if self.is_training:
            '''
            in training. We should show the detecitons in the tensorboard. So we add this.
            '''
            kept_indices = tf.reshape(tf.where(tf.greater_equal(final_scores, cfgs.SHOW_SCORE_THRSHOLD)), [-1])
            final_boxes = tf.gather(final_boxes, kept_indices)
            final_scores = tf.gather(final_scores, kept_indices)
            final_category = tf.gather(final_category, kept_indices)

        return final_boxes, final_scores, final_category

    def roi_pooling(self, feature_maps, rois, img_shape):
        '''
        Here use roi warping as roi_pooling
        only use the 7x7 ROIpooling because the 1080ti OOM
        :param featuremaps_dict: feature map to crop
        :param rois: shape is [-1, 4]. [x1, y1, x2, y2]
        :return:
        '''
        img_h, img_w = tf.cast(img_shape[1], tf.float32), tf.cast(img_shape[2], tf.float32)
        roi_all_features = []
        for level in self.level:
            with tf.variable_scope('ROI_Warping_{}'.format(level)):
                with slim.arg_scope([slim.fully_connected], weights_regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):
                    # tf.cast()函数的作用是用于类型转换
                    N = tf.shape(rois[level])[0]
                    x1, y1, x2, y2 = tf.unstack(rois[level], axis=1)

                    # 将bbox缩放到feature maps上 前面我们得到的rois是映射回输入图像中的
                    normalized_x1 = x1 / img_w
                    normalized_x2 = x2 / img_w
                    normalized_y1 = y1 / img_h
                    normalized_y2 = y2 / img_h

                    normalized_rois = tf.transpose(
                        tf.stack([normalized_y1, normalized_x1, normalized_y2, normalized_x2]), name='get_normalized_rois_{}'.format(level))

                    normalized_rois = tf.stop_gradient(normalized_rois)


                    #for i in range(3):
                    cropped_roi_features = tf.image.crop_and_resize(feature_maps[level], normalized_rois,
                                                                    box_ind=tf.zeros(shape=[N, ],
                                                                                        dtype=tf.int32),
                                                                    crop_size=[cfgs.ROI_SIZE, cfgs.ROI_SIZE],
                                                                    name='CROP_AND_RESIZE_{}'.format(level)
                                                                    )

                    roi_features = slim.max_pool2d(cropped_roi_features,
                                                    [cfgs.ROI_POOL_KERNEL_SIZE, cfgs.ROI_POOL_KERNEL_SIZE],
                                                    stride=cfgs.ROI_POOL_KERNEL_SIZE, padding='SAME')

                    roi_all_features.append(roi_features)
                    # roi_features_ = slim.flatten(roi_features)
                    # roi_all_features.append(roi_features_)

        # roi_features_concat = tf.concat(roi_all_features, axis=1)

        roi_features = tf.concat(roi_all_features, axis=0)

        return roi_features

    def build_fastrcnn(self, feature_to_cropped, rois_all, img_shape):

        with tf.variable_scope('Fast-RCNN'):

            # 5. ROI Pooling
            with tf.variable_scope('rois_pooling'):
                pooled_features = self.roi_pooling(feature_maps=feature_to_cropped, rois=rois_all, img_shape=img_shape)

            #6. inferecne rois in Fast-RCNN to obtain fc_flatten features
            if self.base_network_name.startswith('resnet'):

                fc_flatten = resnet.restnet_head(input=pooled_features,
                                                 is_training=self.is_training,
                                                 scope_name=self.base_network_name) # self.base_network_name
                #fc_flatten = pooled_features

            elif self.base_network_name.startswith('MobilenetV2'):
                fc_flatten = mobilenet_v2.mobilenetv2_head(inputs=pooled_features,
                                                           is_training=self.is_training)
            else:
                raise NotImplementedError('only support resnet and mobilenet')

            # 7. cls and reg in Fast-RCNN
            with tf.variable_scope('horizen_branch'):
                with slim.arg_scope([slim.fully_connected], weights_regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):

                    print('*'*20, fc_flatten.shape)
                    fc6 = slim.fully_connected(fc_flatten, 2048, scope='fc_1')

                    if self.usedropout:
                        fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=self.is_training, scope='dropout_1')


                    fc7 = slim.fully_connected(fc6, 2048, scope='fc_2')
                    if self.usedropout:
                        fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=self.is_training, scope='dropout_2')

                    cls_score_h = slim.fully_connected(fc7,
                                                       num_outputs=cfgs.CLASS_NUM+1,
                                                       weights_initializer=cfgs.INITIALIZER,
                                                       activation_fn=None, trainable=self.is_training,
                                                       scope='cls_fc_h')

                    bbox_pred_h = slim.fully_connected(fc7,
                                                       num_outputs=(cfgs.CLASS_NUM+1) * 4,
                                                       weights_initializer=cfgs.BBOX_INITIALIZER,
                                                       activation_fn=None, trainable=self.is_training,
                                                       scope='reg_fc_h')
                    # for convient. It also produce (cls_num +1) bboxes

                    cls_score_h = tf.reshape(cls_score_h, [-1, cfgs.CLASS_NUM+1])
                    bbox_pred_h = tf.reshape(bbox_pred_h, [-1, 4*(cfgs.CLASS_NUM+1)])

            with tf.variable_scope('rotation_branch'):
                with slim.arg_scope([slim.fully_connected], weights_regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):
                    cls_score_r = slim.fully_connected(fc_flatten,
                                                       num_outputs=cfgs.CLASS_NUM + 1,
                                                       weights_initializer=cfgs.INITIALIZER,
                                                       activation_fn=None, trainable=self.is_training,
                                                       scope='cls_fc_r')

                    bbox_pred_r = slim.fully_connected(fc_flatten,
                                                       num_outputs=(cfgs.CLASS_NUM + 1) * 5,
                                                       weights_initializer=cfgs.BBOX_INITIALIZER,
                                                       activation_fn=None, trainable=self.is_training,
                                                       scope='reg_fc_r')
                    # for convient. It also produce (cls_num +1) bboxes
                    cls_score_r = tf.reshape(cls_score_r, [-1, cfgs.CLASS_NUM + 1])
                    bbox_pred_r = tf.reshape(bbox_pred_r, [-1, 5 * (cfgs.CLASS_NUM + 1)])

            return bbox_pred_h, cls_score_h, bbox_pred_r, cls_score_r

    def add_anchor_img_smry(self, img, anchors, labels):

        positive_anchor_indices = tf.reshape(tf.where(tf.greater_equal(labels, 1)), [-1])
        negative_anchor_indices = tf.reshape(tf.where(tf.equal(labels, 0)), [-1])

        positive_anchor = tf.gather(anchors, positive_anchor_indices)
        negative_anchor = tf.gather(anchors, negative_anchor_indices)

        pos_in_img = show_box_in_tensor.draw_box_with_color(img, positive_anchor, tf.shape(positive_anchor)[0])
        neg_in_img = show_box_in_tensor.draw_box_with_color(img, negative_anchor, tf.shape(positive_anchor)[0])

        tf.summary.image('positive_anchor', pos_in_img)
        tf.summary.image('negative_anchors', neg_in_img)

    def add_roi_batch_img_smry(self, img, rois, labels):
        positive_roi_indices = tf.reshape(tf.where(tf.greater_equal(labels, 1)), [-1])

        negative_roi_indices = tf.reshape(tf.where(tf.equal(labels, 0)), [-1])

        pos_roi = tf.gather(rois, positive_roi_indices)
        neg_roi = tf.gather(rois, negative_roi_indices)

        pos_in_img = show_box_in_tensor.draw_box_with_color(img, pos_roi, tf.shape(pos_roi)[0])
        neg_in_img = show_box_in_tensor.draw_box_with_color(img, neg_roi, tf.shape(neg_roi)[0])

        tf.summary.image('pos_rois', pos_in_img)
        tf.summary.image('neg_rois', neg_in_img)

    def build_loss(self, rpn_box_pred, rpn_bbox_targets, rpn_cls_score, rpn_labels,
                   bbox_pred_h, bbox_targets_h, cls_score_h, bbox_pred_r, bbox_targets_r, cls_score_r, labels):
        '''

        :param rpn_box_pred: [-1, 4]
        :param rpn_bbox_targets: [-1, 4]
        :param rpn_cls_score: [-1]
        :param rpn_labels: [-1]
        :param bbox_pred_h: [-1, 4*(cls_num+1)]
        :param bbox_targets_h: [-1, 4*(cls_num+1)]
        :param cls_score_h: [-1, cls_num+1]
        :param bbox_pred_r: [-1, 5*(cls_num+1)]
        :param bbox_targets_r: [-1, 5*(cls_num+1)]
        :param cls_score_r: [-1, cls_num+1]
        :param labels: [-1]
        :return:
        '''
        rpn_all_bbox_loss = []
        rpn_all_cls_loss = []
        with tf.variable_scope('build_loss') as sc:
            for level in self.level:
                with tf.variable_scope('rpn_loss_{}'.format(level)):

                    rpn_labels1=rpn_labels[level]
                    rpn_bbox_loss_ = losses.smooth_l1_loss_rpn(bbox_pred=rpn_box_pred[level],
                                                              bbox_targets=rpn_bbox_targets[level],
                                                              label=rpn_labels1,
                                                              sigma=cfgs.RPN_SIGMA)
                    # rpn_cls_loss:
                    # rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])
                    # rpn_labels = tf.reshape(rpn_labels, [-1])
                    # ensure rpn_labels shape is [-1]
                    rpn_select = tf.reshape(tf.where(tf.not_equal(rpn_labels1, -1)), [-1])
                    rpn_cls_score1 = tf.reshape(tf.gather(rpn_cls_score[level], rpn_select), [-1, 2])
                    rpn_labels1 = tf.reshape(tf.gather(rpn_labels1, rpn_select), [-1])
                    rpn_cls_loss_ = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score1,
                                                                                                 labels=rpn_labels1))

                    rpn_cls_loss_ = rpn_cls_loss_ * cfgs.RPN_CLASSIFICATION_LOSS_WEIGHT
                    rpn_bbox_loss_ = rpn_bbox_loss_ * cfgs.RPN_LOCATION_LOSS_WEIGHT

                    rpn_all_bbox_loss.append(rpn_bbox_loss_)
                    rpn_all_cls_loss.append(rpn_cls_loss_)

            # 将两个rpn的loss加在一起构成最终的rpn_loss
            rpn_cls_loss = sum(rpn_all_cls_loss)
            rpn_bbox_loss = sum(rpn_all_bbox_loss)
#-----------------------------------FastRCNN_loss--------------------------------------------------------------

            with tf.variable_scope('FastRCNN_loss'):
                if not cfgs.FAST_RCNN_MINIBATCH_SIZE == -1:
                    bbox_loss_h = losses.smooth_l1_loss_rcnn_h(bbox_pred=bbox_pred_h,
                                                               bbox_targets=bbox_targets_h,
                                                               label=labels,
                                                               num_classes=cfgs.CLASS_NUM + 1,
                                                               sigma=cfgs.FASTRCNN_SIGMA)

                    # cls_score = tf.reshape(cls_score, [-1, cfgs.CLASS_NUM + 1])
                    # labels = tf.reshape(labels, [-1])
                    cls_loss_h = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=cls_score_h,
                        labels=labels))  # beacause already sample before

                    bbox_loss_r = losses.smooth_l1_loss_rcnn_r(bbox_pred=bbox_pred_r,
                                                               bbox_targets=bbox_targets_r,
                                                               label=labels,
                                                               num_classes=cfgs.CLASS_NUM + 1,
                                                               sigma=cfgs.FASTRCNN_SIGMA)

                    cls_loss_r = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=cls_score_r,
                        labels=labels))
                else:
                    ''' 
                    applying OHEM here
                    '''
                    print(20 * "@@")
                    print("@@" + 10 * " " + "TRAIN WITH OHEM ...")
                    print(20 * "@@")
                    cls_loss = bbox_loss = losses.sum_ohem_loss(
                        cls_score=cls_score_h,
                        label=labels,
                        bbox_targets=bbox_targets_h,
                        nr_ohem_sampling=128,
                        nr_classes=cfgs.CLASS_NUM + 1)

                cls_loss_h = cls_loss_h * cfgs.FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT
                bbox_loss_h = bbox_loss_h * cfgs.FAST_RCNN_LOCATION_LOSS_WEIGHT
                cls_loss_r = cls_loss_r * cfgs.FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT
                bbox_loss_r = bbox_loss_r * cfgs.FAST_RCNN_LOCATION_LOSS_WEIGHT


            loss_dict = {
                'rpn_cls_loss': rpn_cls_loss,
                'rpn_loc_loss': rpn_bbox_loss,
                'fastrcnn_cls_loss_h': cls_loss_h,
                'fastrcnn_loc_loss_h': bbox_loss_h,
                'fastrcnn_cls_loss_r': cls_loss_r,
                'fastrcnn_loc_loss_r': bbox_loss_r,
            }
        return loss_dict

    def build_whole_detection_network(self, input_img_batch, gtboxes_r_batch, gtboxes_h_batch):

        if self.is_training:
            # ensure shape is [M, 5] and [M, 6]
            gtboxes_r_batch = tf.reshape(gtboxes_r_batch, [-1, 6])
            gtboxes_h_batch = tf.reshape(gtboxes_h_batch, [-1, 5])
            gtboxes_r_batch = tf.cast(gtboxes_r_batch, tf.float32)
            gtboxes_h_batch = tf.cast(gtboxes_h_batch, tf.float32)

        img_shape = tf.shape(input_img_batch)

        # 1. build base network
        C2_, C4 = self.build_base_network(input_img_batch)


        C2 = slim.conv2d(C2_, num_outputs=1024, kernel_size=[1, 1], stride=1,scope='build_C2_to_1024')

        self.feature_pyramid = {'C2':C2, 'C4':C4}


        # 2. build rpn

        rpn_all_encode_boxes = {}
        rpn_all_boxes_scores = {}
        rpn_all_cls_score = {}
        anchors = {}

        with tf.variable_scope('build_rpn',regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):
            i = 0
            for level in self.level:
                rpn_conv3x3 = slim.conv2d(
                    self.feature_pyramid[level], 512, [3, 3],
                    trainable=self.is_training, weights_initializer=cfgs.INITIALIZER,
                    activation_fn=tf.nn.relu,
                    scope='rpn_conv/3x3_{}'.format(level))
                rpn_cls_score = slim.conv2d(rpn_conv3x3, self.num_anchors_per_location[i]*2, [1, 1], stride=1,
                                            trainable=self.is_training, weights_initializer=cfgs.INITIALIZER,
                                            activation_fn=None,
                                            scope='rpn_cls_score_{}'.format(level))
                rpn_box_pred = slim.conv2d(rpn_conv3x3, self.num_anchors_per_location[i]*4, [1, 1], stride=1,
                                           trainable=self.is_training, weights_initializer=cfgs.BBOX_INITIALIZER,
                                           activation_fn=None,
                                           scope='rpn_bbox_pred_{}'.format(level))
                rpn_box_pred = tf.reshape(rpn_box_pred, [-1, 4])
                rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])
                rpn_cls_prob = slim.softmax(rpn_cls_score, scope='rpn_cls_prob_{}'.format(level)) # do the softmax

                rpn_all_cls_score[level]  = rpn_cls_score
                rpn_all_boxes_scores[level] = rpn_cls_prob  # do the softmax
                rpn_all_encode_boxes[level] = rpn_box_pred
                i += 1


        # 3. generate_anchors
        i = 0
        for level, base_anchor_size, stride in zip(self.level, self.base_anchor_size_list, self.stride):
            featuremap_height, featuremap_width = tf.shape(self.feature_pyramid[level])[1], tf.shape(self.feature_pyramid[level])[2]

            featuremap_height = tf.cast(featuremap_height, tf.float32)
            featuremap_width = tf.cast(featuremap_width, tf.float32)

            #anchor_scale = tf.constant(self.anchor_scales[i], dtype=tf.float32)
            #)anchor_ratio = tf.constant(self.anchor_ratios[i], dtype=tf.float32)
            anchor_scale = self.anchor_scales[i]
            anchor_ratio = self.anchor_ratios[i]

            tmp_anchors = anchor_utils.make_anchors(base_anchor_size=base_anchor_size,
                                                anchor_scales=anchor_scale, anchor_ratios=anchor_ratio,
                                                featuremap_height=featuremap_height,
                                                featuremap_width=featuremap_width,
                                                stride=stride,
                                                name="make_anchors_forRPN_{}".format(level))
            tmp_anchors = tf.reshape(tmp_anchors, [-1, 4])
            anchors[level] = tmp_anchors
            i += 1


        # with tf.variable_scope('make_anchors'):
        #     anchors = anchor_utils.make_anchors(height=featuremap_height,
        #                                         width=featuremap_width,
        #                                         feat_stride=cfgs.ANCHOR_STRIDE[0],
        #                                         anchor_scales=cfgs.ANCHOR_SCALES,
        #                                         anchor_ratios=cfgs.ANCHOR_RATIOS, base_size=16
        #                                         )

        # 4. postprocess rpn proposals. such as: decode, clip, NMS
        rois = {}
        roi_scores = {}
        with tf.variable_scope('postprocess_RPN'):
            # rpn_cls_prob = tf.reshape(rpn_cls_score, [-1, 2])
            # rpn_cls_prob = slim.softmax(rpn_cls_prob, scope='rpn_cls_prob')
            # rpn_box_pred = tf.reshape(rpn_box_pred, [-1, 4])
            for level in self.level:
                rois_rpn, roi_scores_rpn = postprocess_rpn_proposals(rpn_bbox_pred=rpn_all_encode_boxes[level],
                                                             rpn_cls_prob=rpn_all_boxes_scores[level],
                                                             img_shape=img_shape,
                                                             anchors=anchors[level],
                                                             is_training=self.is_training)
                # rois[level] = rois
                # roi_scores[level] = roi_scores
            # rois shape [-1, 4]
            # +++++++++++++++++++++++++++++++++++++add img smry+++++++++++++++++++++++++++++++++++++++++++++++++++++++
                rois[level] = rois_rpn
                roi_scores[level] = roi_scores_rpn

                if self.is_training:
                    rois_in_img = show_box_in_tensor.draw_boxes_with_categories(img_batch=input_img_batch,
                                                                                boxes=rois_rpn,
                                                                                scores=roi_scores_rpn)
                    tf.summary.image('all_rpn_rois_{}'.format(level), rois_in_img)

                    score_gre_05 = tf.reshape(tf.where(tf.greater_equal(roi_scores_rpn, 0.5)), [-1])
                    score_gre_05_rois = tf.gather(rois_rpn, score_gre_05)
                    score_gre_05_score = tf.gather(roi_scores_rpn, score_gre_05)
                    score_gre_05_in_img = show_box_in_tensor.draw_boxes_with_categories(img_batch=input_img_batch,
                                                                                        boxes=score_gre_05_rois,
                                                                                        scores=score_gre_05_score)
                    tf.summary.image('score_greater_05_rois_{}'.format(level), score_gre_05_in_img)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


        rpn_labels = {}
        rpn_bbox_targets = {}
        labels_all = []
        labels = {}
        bbox_targets_h = {}
        bbox_targets_r = {}
        bbox_targets_all_h = []
        bbox_targets_all_r = []

        if self.is_training:
            for level in self.level:
                with tf.variable_scope('sample_anchors_minibatch_{}'.format(level)):
                    rpn_labels_one, rpn_bbox_targets_one = \
                        tf.py_func(
                            anchor_target_layer,
                            [gtboxes_h_batch, img_shape, anchors[level]],
                            [tf.float32, tf.float32])
                    rpn_bbox_targets_one = tf.reshape(rpn_bbox_targets_one, [-1, 4])
                    rpn_labels_one = tf.to_int32(rpn_labels_one, name="to_int32_{}".format(level))
                    rpn_labels_one = tf.reshape(rpn_labels_one, [-1])
                    self.add_anchor_img_smry(input_img_batch, anchors[level], rpn_labels_one)

                # -----------------------------add to the dict-------------------------------------------------------------
                    rpn_labels[level] = rpn_labels_one
                    rpn_bbox_targets[level] = rpn_bbox_targets_one
                # --------------------------------------add smry-----------------------------------------------------------

                rpn_cls_category = tf.argmax(rpn_all_boxes_scores[level], axis=1)
                kept_rpppn = tf.reshape(tf.where(tf.not_equal(rpn_labels_one, -1)), [-1])
                rpn_cls_category = tf.gather(rpn_cls_category, kept_rpppn) # 预测
                acc = tf.reduce_mean(tf.to_float(tf.equal(rpn_cls_category, tf.to_int64(tf.gather(rpn_labels_one, kept_rpppn)))))
                tf.summary.scalar('ACC/rpn_accuracy_{}'.format(level), acc)

                with tf.control_dependencies([rpn_labels[level]]):
                    with tf.variable_scope('sample_RCNN_minibatch_{}'.format(level)):
                        rois_, labels_, bbox_targets_h_, bbox_targets_r_ = \
                        tf.py_func(proposal_target_layer,
                                   [rois[level], gtboxes_h_batch, gtboxes_r_batch],
                                   [tf.float32, tf.float32, tf.float32, tf.float32])

                        rois_fast = tf.reshape(rois_, [-1, 4])
                        labels_fast = tf.to_int32(labels_)
                        labels_fast = tf.reshape(labels_fast, [-1])
                        bbox_targets_h_fast = tf.reshape(bbox_targets_h_, [-1, 4*(cfgs.CLASS_NUM+1)])
                        bbox_targets_r_fast = tf.reshape(bbox_targets_r_, [-1, 5*(cfgs.CLASS_NUM+1)])
                        self.add_roi_batch_img_smry(input_img_batch, rois_fast, labels_fast)
                #----------------------new_add----------------------
                        rois[level] = rois_fast
                        labels[level] = labels_fast
                        bbox_targets_h[level] = bbox_targets_h_fast
                        bbox_targets_r[level] = bbox_targets_r_fast
                        labels_all.append(labels_fast)
                        bbox_targets_all_h.append(bbox_targets_h_fast)
                        bbox_targets_all_r.append(bbox_targets_r_fast)

            fast_labels = tf.concat(labels_all, axis=0)
            fast_bbox_targets_h = tf.concat(bbox_targets_all_h, axis=0)
            fast_bbox_targets_r = tf.concat(bbox_targets_all_r, axis=0)
        # -------------------------------------------------------------------------------------------------------------#
        #                                            Fast-RCNN                                                         #
        # -------------------------------------------------------------------------------------------------------------#

        # 5. build Fast-RCNN
        # rois = tf.Print(rois, [tf.shape(rois)], 'rois shape', summarize=10)


        bbox_pred_h, cls_score_h, bbox_pred_r, cls_score_r = self.build_fastrcnn(feature_to_cropped=self.feature_pyramid,
                                                                                 rois_all=rois,
                                                                                 img_shape=img_shape)


        # 这里的feature_to_cropped是feature maps 特征图
        # bbox_pred shape: [-1, 4*(cls_num+1)].
        # cls_score shape： [-1, cls_num+1]

        cls_prob_h = slim.softmax(cls_score_h, 'cls_prob_h') # 根据代码可看到水平和旋转的处理过程是分开的
        cls_prob_r = slim.softmax(cls_score_r, 'cls_prob_r')

        # ----------------------------------------------add smry-------------------------------------------------------
        if self.is_training:

            cls_category_h = tf.argmax(cls_prob_h, axis=1)
            fast_acc_h = tf.reduce_mean(tf.to_float(tf.equal(cls_category_h, tf.to_int64(fast_labels))))
            tf.summary.scalar('ACC/fast_acc_h', fast_acc_h)

            cls_category_r = tf.argmax(cls_prob_r, axis=1)
            fast_acc_r = tf.reduce_mean(tf.to_float(tf.equal(cls_category_r, tf.to_int64(fast_labels))))
            tf.summary.scalar('ACC/fast_acc_r', fast_acc_r)

        #  6. postprocess_fastrcnn
        if not self.is_training:

            rois_all = []
            for level in self.level:
                rois_all.append(rois[level])
            rois = tf.concat(rois_all, axis=0)

            final_boxes_h, final_scores_h, final_category_h = self.postprocess_fastrcnn_h(rois=rois,
                                                                                          bbox_ppred=bbox_pred_h,
                                                                                          scores=cls_prob_h,
                                                                                          img_shape=img_shape)
            final_boxes_r, final_scores_r, final_category_r = self.postprocess_fastrcnn_r(rois=rois,
                                                                                          bbox_ppred=bbox_pred_r,
                                                                                          scores=cls_prob_r,
                                                                                          img_shape=img_shape)
            return final_boxes_h, final_scores_h, final_category_h, final_boxes_r, final_scores_r, final_category_r
        else:
            '''
            when trian. We need build Loss
            '''
            loss_dict = self.build_loss(rpn_box_pred=rpn_all_encode_boxes,
                                        rpn_bbox_targets=rpn_bbox_targets,
                                        rpn_cls_score=rpn_all_cls_score,
                                        rpn_labels=rpn_labels,
                                        bbox_pred_h=bbox_pred_h,
                                        bbox_targets_h=fast_bbox_targets_h,
                                        cls_score_h=cls_score_h,
                                        bbox_pred_r=bbox_pred_r,
                                        bbox_targets_r=fast_bbox_targets_r,
                                        cls_score_r=cls_score_r,
                                        labels=fast_labels)
            rois_all = []
            for level in self.level:
                rois_all.append(rois[level])
            rois = tf.concat(rois_all, axis=0)

            final_boxes_h, final_scores_h, final_category_h = self.postprocess_fastrcnn_h(rois=rois,
                                                                                          bbox_ppred=bbox_pred_h,
                                                                                          scores=cls_prob_h,
                                                                                          img_shape=img_shape)
            final_boxes_r, final_scores_r, final_category_r = self.postprocess_fastrcnn_r(rois=rois,
                                                                                          bbox_ppred=bbox_pred_r,
                                                                                          scores=cls_prob_r,
                                                                                          img_shape=img_shape)

            return final_boxes_h, final_scores_h, final_category_h, \
                   final_boxes_r, final_scores_r, final_category_r, loss_dict

    def get_restorer(self):
        checkpoint_path = tf.train.latest_checkpoint(os.path.join(cfgs.TRAINED_CKPT, cfgs.VERSION))

        if checkpoint_path != None:
            if cfgs.RESTORE_FROM_RPN:
                print('___restore from rpn___')
                model_variables = slim.get_model_variables()
                restore_variables = [var for var in model_variables if not var.name.startswith('FastRCNN_Head')] + \
                                    [slim.get_or_create_global_step()]
                for var in restore_variables:
                    print(var.name)
                restorer = tf.train.Saver(restore_variables)
            else:
                restorer = tf.train.Saver()
            print("model restore from :", checkpoint_path)
        else:
            checkpoint_path = cfgs.PRETRAINED_CKPT
            print("model restore from pretrained mode, path is :", checkpoint_path)

            model_variables = slim.get_model_variables()
            # print(model_variables)

            def name_in_ckpt_rpn(var):
                return var.op.name

            def name_in_ckpt_fastrcnn_head(var):
                '''
                Fast-RCNN/resnet_v1_50/block4 -->resnet_v1_50/block4
                :param var:
                :return:
                '''
                return '/'.join(var.op.name.split('/')[1:])

            nameInCkpt_Var_dict = {}
            for var in model_variables:
                if var.name.startswith('Fast-RCNN/'+self.base_network_name+'/block4'):
                    var_name_in_ckpt = name_in_ckpt_fastrcnn_head(var)
                    nameInCkpt_Var_dict[var_name_in_ckpt] = var
                else:
                    if var.name.startswith(self.base_network_name):
                        var_name_in_ckpt = name_in_ckpt_rpn(var)
                        nameInCkpt_Var_dict[var_name_in_ckpt] = var
                    else:
                        continue
            restore_variables = nameInCkpt_Var_dict
            for key, item in restore_variables.items():
                print("var_in_graph: ", item.name)
                print("var_in_ckpt: ", key)
                print(20*"---")
            restorer = tf.train.Saver(restore_variables)
            print(20 * "****")
            print("restore from pretrained_weighs in IMAGE_NET")
        return restorer, checkpoint_path

    def get_gradients(self, optimizer, loss):
        '''

        :param optimizer:
        :param loss:
        :return:

        return vars and grads that not be fixed
        '''

        # if cfgs.FIXED_BLOCKS > 0:
        #     trainable_vars = tf.trainable_variables()
        #     # trained_vars = slim.get_trainable_variables()
        #     start_names = [cfgs.NET_NAME + '/block%d'%i for i in range(1, cfgs.FIXED_BLOCKS+1)] + \
        #                   [cfgs.NET_NAME + '/conv1']
        #     start_names = tuple(start_names)
        #     trained_var_list = []
        #     for var in trainable_vars:
        #         if not var.name.startswith(start_names):
        #             trained_var_list.append(var)
        #     # slim.learning.train()
        #     grads = optimizer.compute_gradients(loss, var_list=trained_var_list)
        #     return grads
        # else:
        #     return optimizer.compute_gradients(loss)
        return optimizer.compute_gradients(loss)

    def enlarge_gradients_for_bias(self, gradients):

        final_gradients = []
        with tf.variable_scope("Gradient_Mult") as scope:
            for grad, var in gradients:
                scale = 1.0
                if cfgs.MUTILPY_BIAS_GRADIENT and './biases' in var.name:
                    scale = scale * cfgs.MUTILPY_BIAS_GRADIENT
                if not np.allclose(scale, 1.0):
                    grad = tf.multiply(grad, scale)
                final_gradients.append((grad, var))
        return final_gradients




















