# imports
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

# argument parser
arg = argparse.ArgumentParser()
arg.add_argument("--image", required=True, help="path to input image")
arg.add_argument("--face", type=str, default="face_detector", help="path to face detector model directory")
arg.add_argument("--model", type=str, default="classifier.model", help="path to trained face mask detector model")
arg.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(arg.parse_args())

# load face detector model from disk
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# load face mask classifier model
model = load_model(args["model"])

# load input image
image = cv2.imread(args["image"])
orig = image.copy()
(h, w) = image.shape[:2]

# construct blob from image
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

# pass the blob through network and obtain face detections
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
    confidence =detections[0, 0, i, 2]

    if confidence > args["confidence"]:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

        face = image[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        (mask, without_mask) = model.predict(face)[0]
        label = "Mask" if mask > without_mask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(mask, without_mask) * 100)
        cv2.putText(image, label, (startX, startY - 10), cv2. FONT_HERSHEY_SIMPLEX, 0.45, color,2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        print(label)

cv2.imwrite("output_fm.png", image)