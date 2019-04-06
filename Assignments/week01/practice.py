'''
Assignments
1. Recode all examples.
2. Please change image color through YUV space
3. Combine image crop, color shift, rotation and perspective transform together to complete a data augmentation script.

Author: Peng Lei
Date: 2019/4/5 21:41
'''

import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

# Read image
img = cv2.imread('./Assignments/week01/img/timg.jpg')
cv2.imshow('Buildings', img)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
print(img)
print(img.shape)

print(img.dtype)

# image crop
img_crop = img[:200, :300]
cv2.imshow('crop', img_crop)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()

# channels
B, G, R = cv2.split(img)
cv2.imshow('B', B)
cv2.imshow('G', G)
cv2.imshow('R', R)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()

# change color BGR
def random_light_color(img):
    B, G, R = cv2.split(img)

    b_rand = random.randint(-50, 50)
    if b_rand == 0:
        pass
    elif b_rand > 0:
        lim = 255 - b_rand
        B[B > lim] = 255
        B[B <= lim] = (b_rand + B[B <= lim]).astype(img.dtype)
    elif b_rand < 0:
        lim = 0 - b_rand
        B[B < lim] = 0
        B[B >= lim] = (b_rand + B[B >= lim]).astype(img.dtype)
    
    g_rand = random.randint(-50, 50)
    if g_rand == 0:
        pass
    elif g_rand > 0:
        lim = 255 - g_rand
        G[G > lim] = 255
        G[G <= lim] = (g_rand + G[G <= lim]).astype(img.dtype)
    elif g_rand < 0:
        lim = 0 - g_rand
        G[G < lim] = 0
        G[G >= lim] = (g_rand + G[G >= lim]).astype(img.dtype)
    
    r_rand = random.randint(-50, 50)
    if r_rand == 0:
        pass
    elif r_rand > 0:
        lim = 255 - r_rand
        R[R > lim] = 255
        R[R <= lim] = (r_rand + R[R <= lim]).astype(img.dtype)
    elif r_rand < 0:
        lim = 0 < r_rand
        R[R < lim] = 0
        R[R >= lim] = (r_rand + R[R >= lim]).astype(img.dtype)
    
    return cv2.merge((B, G, R))

# change yuv
def random_yuv_color(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    Y, U, V = cv2.split(img_yuv)
    
    y_rand = random.randint(-50, 50)
    if y_rand == 0:
        pass
    elif y_rand > 0:
        lim = 255 - y_rand
        Y[Y > lim] = 255
        Y[Y <= lim] = (y_rand + Y[Y <= lim]).astype(img_yuv.dtype)
    elif y_rand < 0:
        lim = 0 < y_rand
        Y[Y < lim] = 0
        Y[Y >= lim] = (y_rand + Y[Y >= lim]).astype(img_yuv.dtype)
    
    u_rand = random.randint(-50, 50)
    if u_rand == 0:
        pass
    elif u_rand > 0:
        lim = 255 - u_rand
        U[U > lim] = 255
        U[U <= lim] = (u_rand + U[U <= lim]).astype(img_yuv.dtype)
    elif u_rand < 0:
        lim = 0 < u_rand
        U[U < lim] = 0
        U[U >= lim] = (u_rand + U[U >= lim]).astype(img_yuv.dtype)
    v_rand = random.randint(-50, 50)
    if v_rand == 0:
        pass
    elif v_rand > 0:
        lim = 255 - v_rand
        V[V > lim] = 255
        V[V <= lim] = (v_rand + V[V <= lim]).astype(img_yuv.dtype)
    elif v_rand < 0:
        lim = 0 < v_rand
        V[V < lim] = 0
        V[V >= lim] = (v_rand + V[V >= lim]).astype(img_yuv.dtype)
    
    img_merge = cv2.merge((Y, U, V))
    return cv2.cvtColor(img_merge, cv2.COLOR_YUV2BGR)

img_random_color = random_light_color(img)
cv2.imshow('img_random_color', img_random_color)
img_random_yuv = random_yuv_color(img)
cv2.imshow('img_random_yuv', img_random_yuv)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
    

# gamma correction
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(table).astype("uint8")
    return cv2.LUT(image, table)

img_brighter = adjust_gamma(img, 2)
cv2.imshow('img_brighter', img_brighter)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()

# histogram
img_small_brighter = cv2.resize(img_brighter, (img_brighter.shape[0]//2, img_brighter.shape[1]//2))
plt.hist(img_brighter.flatten(), 256, [0, 256], color = 'r')
img_yuv = cv2.cvtColor(img_small_brighter, cv2.COLOR_BGR2YUV)
# equalize the histogram of the U channel
img_yuv[:,:,1] = cv2.equalizeHist(img_yuv[:,:,1])
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
cv2.imshow('Color input image', img_small_brighter)
cv2.imshow('Histogram equalized', img_output)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()



#Rotate Transform
M = cv2.getRotationMatrix2D((img.shape[0]/2, img.shape[1]/2), 25, 1)
img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow('Rotate', img_rotate)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
print(M)

#set M[0][2] = M[1][2] = 0
M[0][2] = M[1][2] = 0
print(M)
img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow('Rotate2', img_rotate)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
# without offset

# scale+rotation+translation = similarity transform
M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 30, 0.5) # center, angle, scale
img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow('Similarity', img_rotate)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()

# Affine Transform
rows, cols, ch = img.shape
pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])
 
M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow('Affine', dst)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()

# perspective transform
def random_warp(img, row, col):
    height, width, channels = img.shape

    # warp:
    random_margin = 60
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))
    return M_warp, img_warp
M_warp, img_warp = random_warp(img, img.shape[0], img.shape[1])
cv2.imshow('Warp', img_warp)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()
	