#import文
import cv2
import numpy as np
import matplotlib.pyplot as plt


#RGBtoYUV処理


#rgb→y
def rgb2y(r,g,b):
    y = 0.257*r +0.504*g +0.098*b +16
    return y

#rgb→Cr(u)
def rgb2u(r,g,b):
    u = -0.148*r -0.291*g +0.439*b +128.0
    return u

#rgb→Cb(v)
def rgb2v(r,g,b):
    v = 0.439*r -0.368*g -0.071*b +128.0
    return v

#YUVからBGRに戻すコード、係数は調べたものを適用
def y2r(y,u,v):
    R = 1.164*(y-16) + 1.596*(u-128)
    return R
def y2g(y,u,v):
    G = 1.164*(y-16) - 0.391*(v-128) - 0.813*(u-128)
    return G
def y2b(y,u,v):
    B = 1.164*(y-16) + 2.018*(v-128)
    return B


#画像取り出し
img = cv2.imread('./img/eye_image_20220822/sub01_reduced_0000.bmp')
g=0

#画像を平均値フィルタ処理
img = cv2.blur(img, ksize=(5, 5))
img = cv2.blur(img, ksize=(5, 5))

#BGRtoYUV
for i in range(img.shape[0]):#縦
    for j in range(img.shape[1]):#横
        for k in range(3):#画素
            if(k == 0):#Y
                img[i,j,k] = rgb2y(img[i,j,2],img[i,j,1],img[i,j,0])
            if(k == 1):#Cr
                img[i,j,k] = rgb2u(img[i,j,2],img[i,j,1],img[i,j,0])
            if(k == 2):#Cb
                img[i,j,k] = rgb2v(img[i,j,2],img[i,j,1],img[i,j,0])




#フィルター処理
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if(i < img.shape[0]-2 and j < img.shape[1]-2):
            g = ( 1 * img[(i-2),(j-2),0]) # [abs(y-2)][abs(x-2)][2]
            + ( 3 * img[(i-1),(j-1),0])   # [abs(y-1)][abs(x-1)][2]
            + (-3 * img[i+1,j+1,0])   # [y+1][x+1][2]
            + (-1 * img[i+2,j+2,0])   # [y+2][x+2][2]
            + ( 1 * img[i,(j-2),0])     # [y][abs(x-2)][2]
            + ( 3 * img[i,(j-1),0])     # [y][abs(x-1)][2]
            + (-3 * img[i,j+1,0])     # [y][x+1][2]
            + (-1 * img[i,j+2,0])     # [y][x+2][2]
        if(g >= 0 and g <= 255):
            img[i,j,0] = g
        if (g > 255):
            img[i,j,0] = 255
        if (g < 0):
            img[i,j,0] = 0

#YUVtoBGR
for i in range(img.shape[0]):#縦
    for j in range(img.shape[1]):#横
        for k in range(3):#画素
            if(k == 0):#b
                img[i,j,k] = y2b(img[i,j,0],img[i,j,1],img[i,j,2])
            if(k == 1):#g
                img[i,j,k] = y2g(img[i,j,0],img[i,j,1],img[i,j,2])
            if(k == 2):#r
                img[i,j,k] = y2r(img[i,j,0],img[i,j,1],img[i,j,2])            


#グレースケール化(モノクロ出力と書いてあったので)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#出力
cv2.imwrite('./img/Edge_filter_testIbin.jpg', img_gray)
