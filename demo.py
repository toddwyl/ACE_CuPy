import cv2
from ACE_cupy import ACE_cpColor
import os

if not os.path.exists('assets'):
    os.makedirs('assets')

img1 = cv2.imread('data/000001.jpg')
img1_enhance = ACE_cpColor(img1)
cv2.imwrite('assets/1_enhance.jpg', img1_enhance)


img2 = cv2.imread('data/002643.jpg')
img2_enhance = ACE_cpColor(img2)
cv2.imwrite('assets/2643_enhance.jpg', img2_enhance)
