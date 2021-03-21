'''
pip uninstall opencv-python
pip uninstall opencv-contrib-python
pip install opencv_python==3.4.2.16 
pip install opencv-contrib-python==3.4.2.16
'''
import cv2
import matplotlib.pyplot as plt 
img1 = cv2. imread('real.png')
gray1 = cv2. cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('gan.png')
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
t0 = cv2.getTickCount()
f = cv2.xfeatures2d.FREAK_create()
brisk=cv2.BRISK_create()
kp1 = brisk.detect(img1)
kp2 = brisk.detect(img2)
kp1, des1 = f.compute(img1, kp1)
kp2, des2 = f.compute(img2, kp2)

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