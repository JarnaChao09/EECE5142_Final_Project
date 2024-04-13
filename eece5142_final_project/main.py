import cv2 as cv
import math
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Height and width that will be used by the model
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

BG_COLOR = (192, 192, 192) # gray
MASK_COLOR = (255, 255, 255) # white


def resize_and_show(name, image, ax):
    ax.imshow(image)
    ax.set_title(name)
    # print(type(image))
    # h, w = image.shape[:2]
    # if h < w:
    #     img = cv.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    # else:
    #     img = cv.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
    # cv.imshow(name, img)
    # cv.waitKey(0) 
    # cv.destroyAllWindows() 

def main():
    images = ["./images/test_posture_seg.jpeg"]

    figure, (ax0, ax1, ax2) = plt.subplots(1, 3)
    figure.tight_layout()

    for image in images:
        img = cv.imread(image)
        print(image)
        resize_and_show(image, img, ax0)
    
    base_options = python.BaseOptions(model_asset_path="./model/pose_landmarker.task", delegate=mp.tasks.BaseOptions.Delegate.CPU)
    options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=mp.tasks.vision.RunningMode.IMAGE, output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)
    for image_file_name in images:
        image = mp.Image.create_from_file(image_file_name)
        # image = cv.imread(image_file_name)
        # image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        detection_result = detector.detect(image)

        print(detection_result)

        segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
        print(segmentation_mask.dtype)
        visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2)
        from skimage import color
        visualized_mask = color.rgb2gray(visualized_mask)

        ax1.imshow(visualized_mask, cmap="gray")
        ax1.set_title("visualized mask")
        # resize_and_show(image_file_name, visualized_mask, ax1)
        print(np.max(visualized_mask))
        print(np.min(visualized_mask))
        print(visualized_mask.dtype)

        from skimage.morphology import skeletonize, thin
        threshold = 0.5
        boolean_mask = visualized_mask >= threshold
        skeleton = skeletonize(boolean_mask)

        # print(skeleton.shape)

        ax2.imshow(skeleton, cmap=plt.cm.gray)
        ax2.set_title("skeleton")

        plt.show()



        # cv.imshow("skeleton", skeleton)

        
    # base_options = python.BaseOptions(model_asset_path="./model/deeplabv3.tflite")
    # options = vision.ImageSegmenterOptions(base_options=base_options, running_mode=mp.tasks.vision.RunningMode.IMAGE, output_category_mask=True)

    # with vision.ImageSegmenter.create_from_options(options) as segmenter:
    #     for image_file_name in images:
    #         image = cv.imread(image_file_name)
    #         image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    #         segmentation_result = segmenter.segment(image)
    #         category_mask = segmentation_result.category_mask

    #         from collections import Counter
    #         print(Counter(category_mask.numpy_view().flatten()))
            
    #         image_data = image.numpy_view()
    #         fg_image = np.zeros(image_data.shape, dtype=np.uint8)
    #         fg_image[:] = MASK_COLOR
    #         bg_image = np.zeros(image_data.shape, dtype=np.uint8)
    #         bg_image[:] = BG_COLOR
            
    #         condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 10
    #         output_image = np.where(condition, image.numpy_view(), bg_image)
            
    #         print(f"Segmentation mask of {image_file_name}:")
    #         resize_and_show(image_file_name, output_image)
