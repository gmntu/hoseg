## Qualitative Results
<iframe width="560" height="315" src="https://www.youtube.com/embed/96pDNfH1yF0" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/VxVRp63EwXA" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Datasets
[Synthetic train set](https://github.com/gmntu/semseg/tree/master/dataset/train_syn) contains 10k images

[Synthetic test set](https://github.com/gmntu/semseg/tree/master/dataset/test_syn) contains 1k images

[Real test set](https://github.com/gmntu/semseg/tree/master/dataset/test_kv2) contains 1k images captured from a Kinect V2 camera

Code snippet for loading and displaying images

```markdown
import cv2
import numpy as np

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

depth = cv2.imread('../dataset/train_syn/depth/0000000.png', cv2.IMREAD_ANYDEPTH)
label = cv2.imread('../dataset/train_syn/label/0000000.png', cv2.IMREAD_GRAYSCALE)

cv2.imshow('depth', cv2.convertScaleAbs(depth, None, 255/1500, 0))
cv2.imshow('label', convertLabel2Color(label))
cv2.waitKey(0)
```


## FCN Models
To be updated

## Realtime Demo using Kinect V2
To be updated


