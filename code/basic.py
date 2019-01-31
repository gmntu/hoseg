import cv2
import numpy as np

name = 'train_syn'
index = 0

def convertLabel2Color(label):
    height, width = label.shape
    label_color = np.zeros((height, width,3), np.uint8)

    label_color[label == 1] = [255,255,0] # Cyan   -> Foreground Note BGR
    label_color[label == 2] = [255,0,0]   # Blue   -> Right hand
    label_color[label == 3] = [0,0,255]   # Red    -> Left hand
    label_color[label == 4] = [255,0,255] # Violet -> Right Arm
    label_color[label == 5] = [0,255,255] # Yellow -> Left Arm
    label_color[label == 6] = [0,255,0]   # Green  -> Obj    
    label_color[label == 7] = [9,127,255] # Orange -> Table    

    return label_color

depth = cv2.imread('../dataset/'+name+'/depth/'+str(index).zfill(7)+'.png', cv2.IMREAD_ANYDEPTH)
label = cv2.imread('../dataset/'+name+'/label/'+str(index).zfill(7)+'.png', cv2.IMREAD_GRAYSCALE)

cv2.imshow('depth', cv2.convertScaleAbs(depth, None, 255/1500, 0))
cv2.imshow('label', convertLabel2Color(label))
cv2.waitKey(0)
