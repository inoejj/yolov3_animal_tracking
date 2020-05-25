import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import sys
import argparse
from src.keras_yolo3.yolo import YOLO, detect_video
from PIL import Image
from timeit import default_timer as timer
from yolo_utils.utils import  detect_object
import pandas as pd
import numpy as np
import random


def yolov3_detection(projectdir,selectmodel):

    scriptdir = os.path.dirname(os.path.realpath(__file__))
    src_path = os.path.join(scriptdir, "src")

    # Set up folder names for default values
    data_folder = projectdir

    image_folder = os.path.join(data_folder, "Source_Images")
    image_test_folder = os.path.join(image_folder, "Test_Images")

    detection_results_folder = os.path.join(image_folder, "Test_Image_Detection_Results")
    detection_results_file = os.path.join(detection_results_folder, "Detection_Results.csv")

    model_classes = os.path.join(data_folder, "data_classes.txt")

    anchors_path = os.path.join(src_path, "keras_yolo3", "model_data", "yolo_anchors.txt")


    save_img = not False

    input_paths = []
    for i in os.listdir(image_test_folder):
        if i.endswith((".jpg" ,".jpeg" ,".png", ".mp4")):
            a = os.path.join(image_test_folder,i)
            input_paths.append(a)


    # Split images and videos
    img_endings = (".jpg", ".jpg", ".png")
    vid_endings = (".mp4", ".mpeg", ".mpg", ".avi")

    input_image_paths = []
    input_video_paths = []
    for item in input_paths:
        if item.endswith(img_endings):
            input_image_paths.append(item)
        elif item.endswith(vid_endings):
            input_video_paths.append(item)

    output_path = os.path.join(image_folder,'Test_Image_Detection_Results')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    gpu_num = 1
    score = 0.25 # threshold to show confidence default to 0.25

    model_path = selectmodel

    # define YOLO detector
    yolo = YOLO(**{"model_path": model_path,"anchors_path": anchors_path, "classes_path": model_classes,
            "score": score, "gpu_num": gpu_num, "model_image_size": (416, 416),})

    # # Make a dataframe for the prediction outputs
    out_df = pd.DataFrame(columns=["image","image_path","xmin","ymin", "xmax", "ymax", "label", "confidence", "x_size", "y_size"])

    # labels to draw on images
    class_file = open(model_classes, "r")
    input_labels = [line.rstrip("\n") for line in class_file.readlines()]
    print("Found {} input labels: {} ...".format(len(input_labels), input_labels))

    postfix = '_out' ## the final output filename oriname + postfix

    if input_image_paths:
        print("Found {} input images: {} ...".format(len(input_image_paths), [os.path.basename(f) for f in input_image_paths[:5]]))
        start = timer()
        text_out = ""

        # This is for images
        for i, img_path in enumerate(input_image_paths):
            print(img_path)
            prediction, image = detect_object(yolo,img_path,save_img=save_img,save_img_path=output_path,postfix=postfix)
            y_size, x_size, _ = np.array(image).shape
            print('@@@@@@@','prediction',prediction)
            print(y_size,x_size)
            for single_prediction in prediction:
                out_df = out_df.append(pd.DataFrame([[os.path.basename(img_path.rstrip("\n")),img_path.rstrip("\n"),]
                            + single_prediction + [x_size, y_size]],
                        columns=["image", "image_path", "xmin", "ymin","xmax", "ymax", "label",
                            "confidence", "x_size","y_size" ]))

        end = timer()

        print("Processed {} images in {:.1f}sec - {:.1f}FPS".format(
                len(input_image_paths), end - start,len(input_image_paths) / (end - start)))
        out_df.to_csv(detection_results_file, index=False)

    # This is for videos
    if input_video_paths:
        print("Found {} input videos: {} ...".format(len(input_video_paths),[os.path.basename(f) for f in input_video_paths[:5]]))
        start = timer()
        for i, vid_path in enumerate(input_video_paths):
            output_videopath = os.path.join(output_path, os.path.basename(vid_path).replace(".", postfix + "."))
            detect_video(yolo, vid_path, output_path=output_videopath, csvpath=output_path)

        end = timer()
        print("Processed {} videos in {:.1f}sec".format(len(input_video_paths), end - start))


    # Close the current yolo session
    yolo.close_session()
