# Covid-19-Face-Mask-and-Social-Distancing-Detection

This project is used to detect face mask and social distancing violations from input image, video and webcam feeds.

## Models and Techniques
Pretrained **Yolov3** is used to detect people

Pretrained **SSD** is used to detect faces

Trained **MobileNetV2** is used as face mask classifier

**Euclidean distance** is used to calculate social distancing violations

## Getting Started
### Prerequisites
* face-detection
* flask
* imutils
* keras
* matplotlib
* opencv-python
* pandas
* scikit-learn
* tensorflow
* opencv-python

## Quickstart (Demo)
Download the pretrained Yolov3 weights using this [link](https://drive.google.com/file/d/1gqdAighUzlkg-ogA8PWRuPfOH0y8OpMI/view?usp=sharing) and save it to the `yolo-coco/` directory

### Docker for Face Mask Detection on Images
Use the Docker image to run the face mask detector microservice 

Pull the docker image using the command **docker pull rakeshraj97/project1:0.0.1**

Run the docker using the command **docker run -p 12000:12000 rakeshraj97/project1:0.0.1**

Ensure working of the microservice using the command **curl http://0.0.0.0:12000/** or open the link **http://0.0.0.0:12000/** in a web browser to use the web application


## Train Face Mask Detector
