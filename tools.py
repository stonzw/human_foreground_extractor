import cv2
import numpy as np
from PIL import Image
import torchvision
import torchvision.transforms as T


def read_image(image_path):
    img = Image.open(image_path).convert('RGB')
    return np.asarray(img)


def get_simple_image_transform():
    transforms = [T.ToTensor()]
    return T.Compose(transforms)


def create_grabcut_mask(image, grabcut_mask):
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    grabcut_mask, _, _ = cv2.grabCut(
        image,
        grabcut_mask,
        None,
        bgd_model,
        fgd_model,
        5,
        cv2.GC_INIT_WITH_MASK
    )
    return np.where((grabcut_mask == 2) | (grabcut_mask == 0), 0, 1).astype(np.uint8)


class ForeGroundExtractor:
    def __init__(self, mrcnn_pre_process, mrcnn_confidence=0.8, grabcut_foreground_confidence=0.8, detect_object_label=1):
        self.mrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.mrcnn.eval()
        self.mrcnn_confidence = mrcnn_confidence
        self.grabcut_foreground = grabcut_foreground_confidence
        self.trans = mrcnn_pre_process
        self.detect_object_label = detect_object_label

    def mrcnn_output2grabcut_input(self, output):
        boxes = output[0]['boxes'].detach().numpy()
        masks = output[0]['masks'].detach().numpy()
        labels = output[0]['labels'].detach().numpy()
        scores = output[0]['scores'].detach().numpy()
        boxes = boxes[(self.mrcnn_confidence < scores) & (labels == self.detect_object_label)].astype(np.uint64)
        masks = masks[(self.mrcnn_confidence < scores) & (labels == self.detect_object_label)]

        grab_mask = np.zeros(masks.shape[2:], np.uint8)
        for b in boxes:
            grab_mask[b[1]:b[3]:, b[0]:b[2]] = cv2.GC_PR_BGD
        for m in masks:
            grab_mask[self.grabcut_foreground < m[0]] = cv2.GC_FGD
        return grab_mask

    def detect_foreground(self, image):
        output = self.mrcnn([self.trans(Image.fromarray(image))])
        grabcut_input = self.mrcnn_output2grabcut_input(output)
        if not (grabcut_input == cv2.GC_FGD).any():
            return np.zeros(image.shape[:2]).astype(np.uint8)
        return create_grabcut_mask(image, grabcut_input)
