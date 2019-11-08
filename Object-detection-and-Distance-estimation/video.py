from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random


import pandas as pd
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--video", dest = "videofile", help = "Video file to run detection on", default = "5.mp4", type = str)
    
    return parser.parse_args()
    
args = arg_parse()
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()



num_classes = 80
classes = load_classes("data/coco.names")



#Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0 
assert inp_dim > 32

#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()


#Set the model in evaluation mode
model.eval()



def write(x, results, df, count):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results
    
    
    idx = count
    df.at[idx, 'xmin'] = float(c1[0])
    df.at[idx, 'ymin'] = float(c1[1])
    df.at[idx, 'xmax'] = float(c2[0])
    df.at[idx, 'ymax'] = float(c2[1])
    
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    df.at[idx, 'label'] = label
    

#    color = random.choice(colors)
#    cv2.rectangle(img, c1, c2,color, 1)
#    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
#    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
#    cv2.rectangle(img, c1, c2,color, -1)
#    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img


#Detection phase

videofile = args.videofile #or path to the video file. 

cap = cv2.VideoCapture(videofile)  

codec = cap.get(cv2.CAP_PROP_FOURCC)
fps = cap.get(cv2.CAP_PROP_FPS)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
out = cv2.VideoWriter('video_output.mp4', int(codec), int(fps), (int(width), int(height)))

assert cap.isOpened(), 'Cannot capture source'

frames = 0  
start = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:   
        img = prep_image(frame, inp_dim)
#        cv2.imshow("a", frame)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1,2)   
                     
        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()
        
        with torch.no_grad():
            output = model(Variable(img, volatile = False), CUDA)
        output = write_results(output, confidence, num_classes, nms_conf = nms_thesh)


        if type(output) == int:
            frames += 1
            print("FPS of the video is {:5.4f}".format( frames / (time.time() - start)))
            cv2.imshow("frame", frame)
            out.write(frame)
            key = cv2.waitKey(1)
            
            if key & 0xFF == ord('q'):
                break
            continue
        

        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(416/im_dim,1)[0].view(-1,1)
        
        output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
        output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
        
        output[:,1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
        

        classes = load_classes('data/coco.names')
        colors = pkl.load(open("pallete", "rb"))
        
        
        df = pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax',])
        count = 0
#        list(map(lambda x: write(x, frame, df, count), output))
        for x in output:
            write(x, frame, df, count)
            count += 1
        df.to_csv("CSVFile.csv", index=False)
        
        MODEL = "model@1535470106"
        WEIGHTS = "model@1535470106"
        
        df = pd.read_csv("CSVFile.csv")
        X_test = df[['xmin', 'ymin', 'xmax', 'ymax']].values
        y_test = np.array([[10],[25],[45],[70],[100],[135]], dtype=np.float64)
        y_test.reshape(-1,1)
    
        # standardized data
        scalar = StandardScaler()
        X_test = scalar.fit_transform(X_test)
        y_test = scalar.fit_transform(y_test)

        # load json and create model
        json_file = open('KITTI-distance-estimation/distance-estimator/generated_files/{}.json'.format(MODEL), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json( loaded_model_json )
    
        # load weights into new model
        loaded_model.load_weights("KITTI-distance-estimation/distance-estimator/generated_files/{}.h5".format(WEIGHTS))
    
        # evaluate loaded model on test data
        loaded_model.compile(loss='mean_squared_error', optimizer='adam')
        y_pred = loaded_model.predict(X_test)
    
#            # scale up predictions to original values
        y_pred = scalar.inverse_transform(y_pred)
        y_test = scalar.inverse_transform(y_test)
            
        # save predictions
        
        for idx, row in df.iterrows():
            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            x2 = int(row['xmax'])
            y2 = int(row['ymax'])
            label = str(row['label'])
            y_pred_val = round(float(((y_pred[idx]))),1)
            if y_pred_val < 0:
                y_pred_val = 0
                
            if label=='car':
                color = colors[0]
            elif label=='truck':
                color = colors[1]
            elif label=='bus':
                color = colors[2]
            elif label=='train':
                color = colors[3]
            elif label=='bicycle':
                color = colors[4]
            elif label=='motorbike':
                color = colors[5]
            elif label=='aeroplane':
                color = colors[6]
            elif label=='person':
                color = colors[7]
            elif label=='traffic light':
                color = colors[8]
            elif label=='stop sign':
                color = colors[98]
            else:
                color = [255,50,50]
            cv2.rectangle(frame, (x1,y1), (x2,y2),color, 1)
            if label=='car' or label=='truck' or label=='bus' or label=='bicycle' or label=='motorbike' or label=='person':
                string = "{0}".format(label) + " {}m".format(y_pred_val)
            else:
                string = "{0}".format(label) 
            t_size = cv2.getTextSize(string, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
            c2 = x1 + t_size[0] + 3, y1 + t_size[1] + 4
            cv2.rectangle(frame, (x1,y1), c2,color, -1)
            cv2.putText(frame, string, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
            
        frame = cv2.resize(frame, (1280,720)) 
        cv2.imshow("frame", frame)
        out.write(frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        frames += 1
        print(time.time() - start)
        print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
        
        
    else:
        break     

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()





