# upload image: endpoints
# save image
# make prediction
# show results
import os
from flask import Flask
from flask import request
from flask import render_template
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2
import argparse
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = "static/"
#OUTPUT_FOLDER = "/home/rocky/Downloads/project1/static/output/"



# load face detector model from disk
prototxtPath = "Models/face_detector/deploy.prototxt"
weightsPath = "Models/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# load face mask classifier model
model = load_model("Models/classifier.model")

def predict(image_path):
    image = cv2.imread(image_path)
    orig = image.copy()
    (h, w) = image.shape[:2]

    # construct blob from image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through network and obtain face detections
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
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
            cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    cv2.imwrite(image_path, image)

    return label


@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            pred = predict(image_location)
            return render_template("templates/index.html", prediction=pred, image_loc=image_file.filename)
    return render_template("templates/index.html", prediction=0, image_loc=None)

if __name__=="__main__":
    app.run(port=12000, debug=True)