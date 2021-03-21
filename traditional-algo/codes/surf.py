import numpy as np
import cv2
from matplotlib import pyplot as plt

imgname1 = 'real.png'
imgname2 = 'gan.png'

surf = cv2.xfeatures2d.SURF_create()

img1 = cv2.imread(imgname1)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) #灰度处理图像
kp1, des1 = surf.detectAndCompute(img1,None)   #des是描述子

img2 = cv2.imread(imgname2)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)#灰度处理图像
kp2, des2 = surf.detectAndCompute(img2,None)  #des是描述子

# BFMatcher解决匹配
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
# 调整ratio
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
img3 = cv2.cvtColor(img3, cv2. COLOR_BGR2RGB)
plt. imshow(img3),plt.show()