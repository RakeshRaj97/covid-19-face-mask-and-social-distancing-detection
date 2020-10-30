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

Create a virtual environment and install the required dependencies using the command `pip install -r requirements.txt`

#### Face mask detection on images
Test the face mask classifier using the command `python detect_face_mask_image.py --image input/image/path`

The output image is stored as `output_fm.png`

You can also use the flask web app using the command `python app.py` and the application runs in the address http://0.0.0.0:12000/

#### Social distancing detection on images
Test the social distancing detection on images using the command `python detect_social_distance_image.py --image input/image/path --distance [default=100.0]`

Experiment `--distance` value for different images. The output is stored as `output_sd.jpg`

### Face mask and social distancing detection on videos/webcam
Use the command `python video.py --video input/video/path --distance [default=100.0] --frames [default=20]` to test on video files

Experiment `--distance` value for different video files and `--frames` to skip frames. The result frames are stored in `result_frames/` directory

Use the command `python webcam.py` to test using a webcam device

### Docker for Face Mask Detection on Images
Use the [Docker image](https://hub.docker.com/r/rakeshraj97/project1/tags) to run the face mask detector microservice 

Pull the docker image using the command **docker pull rakeshraj97/project1:0.0.1**

Run the docker using the command **docker run -p 12000:12000 rakeshraj97/project1:0.0.1**

Ensure working of the microservice using the command **curl http://0.0.0.0:12000/** or open the link **http://0.0.0.0:12000/** in a web browser to use the web application


## Train Face Mask Detector
### Dataset description
The dataset used to train face mask detector can be downloaded using this [link](https://drive.google.com/drive/folders/1IPwsC30wNAc74_GTXuEWX_F8m2n-ZBCH)

This is a balanced dataset containing faces with and without masks with a mean height of 283.68 and mean width of 278.77

![data](https://user-images.githubusercontent.com/47710229/97522777-a243d080-19f4-11eb-93c9-04dea6ceec6c.png)

### Train
Use the command `python train_mask_detector.py --dataset input/dataset/path` to train the face mask classifier

![plot](https://user-images.githubusercontent.com/47710229/97524430-c0abcb00-19f8-11eb-8543-816514e222f7.png)

## Output of the Trained Model
### Face Mask detection
![output_fm](https://user-images.githubusercontent.com/47710229/97523450-77f31280-19f6-11eb-8ea4-b8c7fa3f849a.png)

### Social Distancing detection
![output_sd](https://user-images.githubusercontent.com/47710229/97523489-90fbc380-19f6-11eb-90f5-864376aaaeed.jpg)

#### References
*https://www.pyimagesearch.com/2020/06/01/opencv-social-distancing-detector/

*https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/

