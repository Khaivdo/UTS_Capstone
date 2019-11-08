# Advanced Driver Assistant Systems
The advanced driver assistance system project is designed to help drivers in daily driving
routines. Technically, the system uses a camera equipped on the car and a human-machine
interface to carry out Object Detection (OD) tasks. With the developed technologies, relevant
objects can be recognised and located. Drivers then will be alerted from potential dangers such 
as car crashes, off-road driving, or they will be assisted in lane-changing situations and high-
way driving, which increases safety for drivers and people around.

## Installation
This software has only been tested on ubuntu 16.04(x64), python3.7, cuda-9.0, cudnn-7.0 with a GTX-1070 GPU. 
The tensorflow version is 1.13.1

## Pretrained Models
Downloads these pretrained models and put them in their corresponding folders before you run the project:

[Lanenet Model](https://www.dropbox.com/sh/tnsf0lw6psszvy4/AAA81r53jpUI3wLsRW6TiPCya?dl=0) - put in /lanenet-lane-detection/model/tusimple_lanenet_vgg

[YOLOv3 Model](https://github.com/ayooshkathuria/pytorch-yolo-v3) - put in /Object-detection-and-Distance-estimation

## Test models
You can test a video on the trained models as follows

### Lane detection
```
cd lanenet-lane-detection

python test_lanenet_video.py --video_path data/tusimple_test_image/5.mp4
```

### Object detection and Distance estimation
```
cd Object-detection-and-Distance-estimation

python video.py --video 5.mp4
```


The final result:

![](https://github.com/Khaivdo/UTS_Capstone/blob/master/Object-detection-and-Distance-estimation/finalResult.png)

While YOLOv3 model was trained on COCO dataset and able to detect up to 80 objects, for the purpose of this project, only the distance between 6 objects below and the camera would be detected:

- Car
- Truck
- Motorbike
- Bicycle
- Bus
- Person


## References
 
[Lanenet-lane-detection](https://github.com/MaybeShewill-CV/lanenet-lane-detection)

[YOLO v3 Object Detector](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)

[KITTI Distance Estimation](https://github.com/harshilpatel312/KITTI-distance-estimation)

[Driving Road Trip Into Sydney](https://www.youtube.com/watch?v=3uShcm7xjq8&t=192s)
