import cv2
import torch, torchvision

import dlib

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import matplotlib.pyplot as plt


def get_detectron_masks(image, predictor, classes=None, expansion=(1., 1.), debug=False):

    # run detectron
    H, W, C = image.shape

    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    labels = instances.pred_classes.numpy()
    seg_masks = instances.pred_masks.numpy()
    boxes = instances.pred_boxes.tensor.numpy()

    # get masks
    if classes is not None:
        indices = [i for i in range(len(instances)) if labels[i] in classes]
        seg_masks = seg_masks[indices]

    ew, eh = expansion
    boxes = np.round(boxes).astype(np.int)
    box_masks = np.zeros([len(instances), H, W], dtype=np.bool)

    for i in range(len(instances)):

        if classes is not None and labels[i] not in classes: continue

        x1, y1, x2, y2 = boxes[i]

        width = x2 - x1
        height = y2 - y1
        dw = int(round((ew - 1.) * width / 2.))
        dh = int(round((eh - 1.) * height / 2.))

        x1 = max(0, x1 - dw)
        x2 = min(W - 1, x2 + dw)
        y1 = max(0, y1 - dh)
        y2 = min(W - 1, y2 + dh)

        box_masks[i, y1:y2, x1:x2] = True

    if debug:
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        seg_mask = v.draw_instance_predictions(instances)
        plt.imshow(seg_mask.get_image())
        plt.show()

    return box_masks, seg_masks


def get_dlib_masks(image, detector, expansion=(2, 1.5)):

    H, W, C = image.shape
    eh, ew = expansion

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(image, 0)
    mask = np.zeros([len(rects), H, W], dtype=np.bool)

    for i, rect in enumerate(rects):

        rect = rect.rect
        width = rect.right() - rect.left()
        height = rect.bottom() - rect.top()
        dw = int(round((ew - 1.) * width / 2.))
        dh = int(round((eh - 1.) * height / 2.))

        x1 = max(0, rect.left() - dw)
        x2 = min(W-1, rect.right() + dw)
        y1 = max(0, rect.top() - dh)
        y2 = min(W-1, rect.bottom() + dh)

        mask[i, y1:y2, x1:x2] = True

    return mask


def get_overlay_mask(image, mask, weight=0.3):

    mask = 255. * np.expand_dims(mask, axis=-1).astype(np.float32)
    mask = np.pad(mask, ((0, 0), (0, 0), (1, 1)), "constant")
    out = weight * image + (1 - weight) * mask
    out = np.round(out).astype(np.uint8)
    return out


def get_face_masks(image,
                   cfg_name="COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
                   dat_path="data/mmod_human_face_detector.dat"):

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(cfg_name))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_name)
    predictor = DefaultPredictor(cfg)

    # dlib_detector = dlib.get_frontal_face_detector()
    dlib_detector = dlib.cnn_face_detection_model_v1(dat_path)

    _, seg_masks = get_detectron_masks(image, predictor)
    seg_mask = seg_masks.sum(axis=0) > 0
    box_masks = get_dlib_masks(image, dlib_detector)

    return seg_mask, box_masks


def get_object_masks(image, classes,
                   cfg_name="COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
                   dat_path="data/mmod_human_face_detector.dat"):

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(cfg_name))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_name)
    predictor = DefaultPredictor(cfg)

    box_masks, seg_masks = get_detectron_masks(image, predictor, classes)
    seg_mask = seg_masks.sum(axis=0) > 0

    return seg_mask, box_masks



if __name__ == "__main__":

    name = "1_97"
    out_dir = "results/stereographic/{}".format(name)
    image = cv2.imread("data/{}.jpg".format(name))

    os.makedirs(out_dir, exist_ok=True)

    cfg_name = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    # load detectron model
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(cfg_name))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_name)
    detectron = DefaultPredictor(cfg)

    dlib_detector = dlib.cnn_face_detection_model_v1("data/mmod_human_face_detector.dat")

    box_masks, seg_masks = get_detectron_masks(image, detectron, classes=[0, 1, 2, 3, 16])
    box_masks = get_dlib_masks(image, dlib_detector)
    box_mask = box_masks.sum(axis=0) > 0
    seg_mask = seg_masks.sum(axis=0) > 0
    # joint_mask = np.logical_and(box_masks, seg_masks).sum(axis=0) > 0
    joint_mask = np.logical_and(box_mask, seg_mask)

    plt.figure(figsize=(10, 8))
    plt.imshow(get_overlay_mask(image, box_mask)[:, :, ::-1])


    plt.figure(figsize=(10, 8))
    plt.imshow(get_overlay_mask(image, seg_mask)[:, :, ::-1])
    plt.show()
    plt.figure(figsize=(10, 8))
    plt.imshow(get_overlay_mask(image, joint_mask)[:, :, ::-1])
    plt.show()