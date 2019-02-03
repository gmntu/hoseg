## Qualitative Results
Video is generated using the sample code (Real-time demo using Kinect V2)
<iframe width="560" height="560" src="https://www.youtube.com/embed/SNuUrp2QiqY" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Datasets
1) [Synthetic train set](https://github.com/gmntu/semseg/tree/master/dataset/train_syn) contains 10k images

2) [Synthetic train set (Fixed body shape)](https://github.com/gmntu/semseg/tree/master/dataset/train_fixbody_syn) contains 10k images

3) [Synthetic test set](https://github.com/gmntu/semseg/tree/master/dataset/test_syn) contains 1k images

4) [Real test set](https://github.com/gmntu/semseg/tree/master/dataset/test_kv2) contains 1k images captured from a Kinect V2 camera

The Synthetic train set is used to train the FCN model.
The Synthetic test set and Real test set are for evaluation.
The Synthetic train set (Fixed body shape) is similar to the Synthetic train set except that all the human models have the same body shape. Its main purpose is to compare and show the improvement in FCN performance when the FCN is trained on 1) Synthetic train set with varying body shapes.

Code snippet for loading and displaying images

```markdown
import cv2
import numpy as np

def convertLabel2Color(label):
	height, width = label.shape
	label_color = np.zeros((height, width,3), np.uint8)
	label_color[label == 1] = [255,255,0] # Cyan   -> Foreground Note BGR
	label_color[label == 2] = [255,0,0]   # Blue   -> Left hand
	label_color[label == 3] = [0,0,255]   # Red    -> Right hand
	label_color[label == 4] = [255,0,255] # Violet -> Left Arm
	label_color[label == 5] = [0,255,255] # Yellow -> Right Arm
	label_color[label == 6] = [0,255,0]   # Green  -> Obj    
	label_color[label == 7] = [9,127,255] # Orange -> Table    
	return label_color

depth = cv2.imread('../dataset/train_syn/depth/0000000.png', cv2.IMREAD_ANYDEPTH)
label = cv2.imread('../dataset/train_syn/label/0000000.png', cv2.IMREAD_GRAYSCALE)

cv2.imshow('depth', cv2.convertScaleAbs(depth, None, 255/1500, 0))
cv2.imshow('label', convertLabel2Color(label))
cv2.waitKey(0)
```

## FCN Models
To be updated

## Real-time Demo using Kinect V2
To be updated


