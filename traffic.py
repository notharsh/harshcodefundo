import os
import logging
import logging.handlers
import random

import numpy as np

import cv2
import matplotlib.pyplot as plt

import utils
cv2.ocl.setUseOpenCL(False)
random.seed(123)


from pipeline import (
    PipelineRunner,
    ContourDetection,
    Visualizer,
    CsvWriter,
    VehicleCounter)

IMAGE_DIR = "./out"
VIDEO_SOURCE = "main1.mp4"
SHAPE = (720, 1280)  #Video Res
EXIT_PTS = np.array([
    [[732, 720], [732, 590], [1280, 500], [1280, 720]],
    [[0, 400], [645, 400], [645, 0], [0, 0]]  
])#Quadrilaterals defining exit areas for both the left and right lanes

def train_bg_subtractor(inst, cap, num=500):
    print ('Training BG Subtractor...')
    i = 0
    while cap.isOpened():
        ret,frame  = cap.read()
        inst.apply(frame, None, 0.001)
        i += 1
        if i >= num:
            return frame


def main():
    log = logging.getLogger("main")


    base = np.zeros(SHAPE + (3,), dtype='uint8')
    exit_mask = cv2.fillPoly(base, EXIT_PTS, (255, 255, 255))[:, :, 0]


    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, detectShadows=True)

    pipeline = PipelineRunner(pipeline=[
        ContourDetection(bg_subtractor=bg_subtractor,
                         save_image=True, image_dir=IMAGE_DIR),
        # y_weight == 2.0 vertical
        # x_weight == 2.0 for horizontal. in below line
        VehicleCounter(exit_masks=[exit_mask], y_weight=2.0),
        Visualizer(image_dir=IMAGE_DIR),
        CsvWriter(path='./', name='report.csv')
    ], log_level=logging.DEBUG)

    cap =cv2.VideoCapture(VIDEO_SOURCE)


    train_bg_subtractor(bg_subtractor, cap, num=500)

    _frame_number = -1
    frame_number = -1
    while cap.isOpened():
        ret,frame = cap.read()

        _frame_number += 1


        if _frame_number % 2 != 0:
            continue

       
        frame_number += 1

        # plt.imshow(frame)
        # plt.show()
        # return

        pipeline.set_context({
            'frame': frame,
            'frame_number': frame_number,
        })
        pipeline.run()
    print("End of Video Reached......")


if __name__ == "__main__":
    log = utils.init_logging()

    if not os.path.exists(IMAGE_DIR):
        log.debug("Creating image directory `%s`...", IMAGE_DIR)
        os.makedirs(IMAGE_DIR)

    main()
