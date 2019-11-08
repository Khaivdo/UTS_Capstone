# YOLO_v3_tutorial_from_scratch
Accompanying code for Paperspace tutorial series ["How to Implement YOLO v3 Object Detector from Scratch"](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)


# Distance Estimation

Originally model was developed by [harshilpatel312](https://github.com/harshilpatel312/KITTI-distance-estimation) which could estimate the distance of an object to the camera in a single frame. 

In this project, I modified the code to detect multiple objects in a video. While YOLOv3 model was trained on COCO dataset and able to detect up to 80 objects, only the distance between 6 objects below and the camera would be detected:

- Car
- Truck
- Motorbike
- Bicycle
- Bus
- Person
