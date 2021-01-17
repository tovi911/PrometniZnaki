import os
import sys
import argparse
from Training.src.keras_yolo3.yolo import YOLO
from PIL import Image
from timeit import default_timer as timer
from Utils.utils import load_extractor_model, load_features, parse_input, detect_object
import test
import Utils.utils
import pandas as pd
import numpy as np
from Utils.Get_File_Paths import GetFileList
import random
from Utils.Train_Utils import get_anchors

yolo_tresh = 0.3

current_path = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(current_path, "Data")
src_path = os.path.join(current_path, "Training", "src")
image_folder = os.path.join(data_folder, "Source_Images")
image_test_folder = os.path.join(image_folder, "Test_Images")
detection_results_folder = os.path.join(image_folder, "Test_Image_Detection_Results")
detection_results_file = os.path.join(detection_results_folder, "Detection_Results.csv")
model_folder = os.path.join(data_folder, "Model_Weights")
model_weights = os.path.join(model_folder, "PrometniZnaki.h5")
model_classes = os.path.join(model_folder, "data_classes.txt")
anchors_path = os.path.join(src_path, "keras_yolo3", "model_data", "yolo_anchors.txt")

# dobi vse slike za test
input_paths = GetFileList(image_test_folder)
# Loči slike od video
img_endings = (".jpg", ".jpeg", ".png")
vid_endings = (".mp4", ".mpeg", ".mpg", ".avi")
input_image_paths = []
input_video_paths = []
for item in input_paths:
    if item.endswith(img_endings):
        input_image_paths.append(item)
    elif item.endswith(vid_endings):
        input_video_paths.append(item)

output_path = detection_results_folder
if not os.path.exists(output_path):
        os.makedirs(output_path)

# Yolo model
yolo = YOLO(**{
    "model_path": model_weights,
    "anchors_path": anchors_path,
    "classes_path": model_classes,
    "score":yolo_tresh,
    "gpu_num":1,
    "model_image_size": (416, 416)})
# Model za izhod
out_df = pd.DataFrame(
        columns=[
            "image",
            "image_path",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "label",
            "confidence",
            "x_size",
            "y_size",])

# Preberi vse classe
class_file = open(model_classes, "r")
input_labels = [line.rstrip("\n") for line in class_file.readlines()]
print("Found {} input labels: {} ...".format(len(input_labels), input_labels))

# Yolo prepoznava
if input_image_paths:
    print("Found {} input images: {} ...".format(len(input_image_paths),[os.path.basename(f) for f in input_image_paths[:5]],))
    start = timer()
    text_out = ""
    # Zanka za izvajanje detekcije na vseh slikah
    for i, img_path in enumerate(input_image_paths):
        img_name = os.path.basename(img_path.rstrip("\n"))
        # Naredi predikcije na trenutni sliki (save_img=1 označi in shrani sliko v output folder!)
        prediction, image = detect_object(yolo,img_path,save_img=0,save_img_path=detection_results_folder)
        y_size, x_size, _ = np.array(image).shape

        print("Na sliki {} je bilo zaznanih {} prometnih znakov\n".format(img_name, len(prediction)))

        # Zanka za ustvarjanje datafraima za vsako predikcijo na trenutni sliki
        for single_prediction in prediction:   
            out_df = out_df.append(pd.DataFrame([[img_name, img_path.rstrip("\n"),]
                        + single_prediction
                        + [x_size, y_size]
                    ],
                    columns=[
                        "image",
                        "image_path",
                        "xmin",
                        "ymin",
                        "xmax",
                        "ymax",
                        "label",
                        "confidence",
                        "x_size",
                        "y_size",
                    ],))
            if single_prediction[5] > yolo_tresh:
               print("");

    end = timer()
    print("Processed {} images in {:.1f}sec - {:.1f}FPS".format(len(input_image_paths), end - start, len(input_image_paths) / (end - start)))
    out_df.to_csv(detection_results_file, index=False)

    yolo.close_session()
