import os
import sys
import argparse
from Training.src.keras_yolo3.yolo import YOLO
from PIL import Image, ImageDraw, ImageFont
from timeit import default_timer as timer
from Utils.utils import load_extractor_model, load_features, parse_input, detect_object
import test
import Utils.utils
import pandas as pd
import numpy as np
from Utils.Get_File_Paths import GetFileList
import random
from Utils.Train_Utils import get_anchors

from keras.models import load_model


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

#model klasifikacije
klas_model = load_model('Data/Model_Weights/klas_prometni.h5')

#klas
classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)', 
            3:'Speed limit (50km/h)', 
            4:'Speed limit (60km/h)', 
            5:'Speed limit (70km/h)', 
            6:'Speed limit (80km/h)', 
            7:'End of speed limit (80km/h)', 
            8:'Speed limit (100km/h)', 
            9:'Speed limit (120km/h)', 
            10:'No passing', 
            11:'No passing veh over 3.5 tons', 
            12:'Right-of-way at intersection', 
            13:'Priority road', 
            14:'Yield', 
            15:'Stop', 
            16:'No vehicles', 
            17:'Veh > 3.5 tons prohibited', 
            18:'No entry', 
            19:'General caution', 
            20:'Dangerous curve left', 
            21:'Dangerous curve right', 
            22:'Double curve', 
            23:'Bumpy road', 
            24:'Slippery road', 
            25:'Road narrows on the right', 
            26:'Road work', 
            27:'Traffic signals', 
            28:'Pedestrians', 
            29:'Children crossing', 
            30:'Bicycles crossing', 
            31:'Beware of ice/snow',
            32:'Wild animals crossing', 
            33:'End speed + passing limits', 
            34:'Turn right ahead', 
            35:'Turn left ahead', 
            36:'Ahead only', 
            37:'Go straight or right', 
            38:'Go straight or left', 
            39:'Keep right', 
            40:'Keep left', 
            41:'Roundabout mandatory', 
            42:'End of no passing', 
            43:'End no passing veh > 3.5 tons' }

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

def classify(image):
    global label_packed
    cimage1 = image.resize((30,30))
    cimage1 = np.expand_dims(cimage1, axis=0)
    cimage1 = np.array(cimage1)
    pred = klas_model.predict_classes([cimage1])[0]
    sign = classes[pred+1]
    print(sign + '\n')
    return sign



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

        try:
            image = Image.open(img_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            image_array = np.array(image)
        except:
            print("File Open Error! Try again!")

        print("Na sliki {} je bilo zaznanih {} prometnih znakov\n".format(img_name, len(prediction)))
        labels = []
        count = 0
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
                cimage  =  image.crop((single_prediction[0]-3, single_prediction[1]-3, single_prediction[2]+3, single_prediction[3]+3))
                label = classify(cimage)
                img_name_test = img_name.replace(".jpg", "")
                cimage.save(os.path.join(detection_results_folder, "Croped_Out", (img_name_test + "_" + str(count) + ".jpg")))
                labels.append(label)
                count = count +1
                


        j = 0
        for single_prediction in prediction:
            if single_prediction[5] > yolo_tresh:
                image_edit = ImageDraw.Draw(image)
                image_edit.rectangle((single_prediction[0]-15, single_prediction[1]-15, single_prediction[2]+15, single_prediction[3]+15), outline = "red", width = 3)
                font = ImageFont.truetype("arial.ttf", size = 30)
                image_edit.text((single_prediction[0]-15, single_prediction[1]-35), labels[j], fill = (0,255,0,0), font = font, stroke_width = 1)
                j = j + 1
        

        image.show()
        output_image_path = os.path.join(detection_results_folder, img_name)
        image.save(output_image_path)




    end = timer()
    print("Processed {} images in {:.1f}sec - {:.1f}FPS".format(len(input_image_paths), end - start, len(input_image_paths) / (end - start)))
    out_df.to_csv(detection_results_file, index=False)

    yolo.close_session()
