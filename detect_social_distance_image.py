# program to compute SD in images
import numpy as np
import argparse
import time
import cv2
import math

# argument parser
arg = argparse.ArgumentParser()
arg.add_argument("--image", required=True, help="path to input image")
arg.add_argument("-d", "--distance", type=float, default=100.0, help="pixels distance to calculate social distance")
args = vars(arg.parse_args())

labelsPath = "./coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")
weightsPath = "./yolo-coco/yolov3.weights"
configPath = "./yolo-coco/yolov3.cfg"

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

image = cv2.imread(args["image"])
(H, W) = image.shape[:2]
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()
print("Frame Prediction Time : {:.6f} seconds".format(end - start))
boxes = []
confidences = []
classIDs = []
for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > 0.5 and classID == 0:
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
ind = []
for i in range(0, len(classIDs)):
    if (classIDs[i] == 0):
        ind.append(i)
a = []
b = []
color = (0, 255, 0)
if len(idxs) > 0:
    for i in idxs.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        a.append(x)
        b.append(y)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

distance = []
nsd = []
for i in range(0, len(a) - 1):
    for k in range(1, len(a)):
        if (k == i):
            break
        else:
            x_dist = (a[k] - a[i])
            y_dist = (b[k] - b[i])
            d = math.sqrt(x_dist * x_dist + y_dist * y_dist)
            distance.append(d)
            if (d <= args["distance"]):
                nsd.append(i)
                nsd.append(k)
            nsd = list(dict.fromkeys(nsd))

color = (0, 0, 255)
for i in nsd:
    (x, y) = (boxes[i][0], boxes[i][1])
    (w, h) = (boxes[i][2], boxes[i][3])
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    text = "Alert"
    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
cv2.imshow("Image", image)
cv2.imwrite('./output_sd.jpg', image)
cv2.waitKey()
