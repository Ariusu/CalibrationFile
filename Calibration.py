#import文
import cv2
import numpy as np
import matplotlib.pyplot as plt

#画像の画素の状態を調べるコード
#print(img[0, :, :].shape)
#print(img[0, :, :])
#print(img[:, 0, :].shape)
#print(img[:, 0, :])
#print(img[:, :, 0].shape)
#print(img[:, :, 0])

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

#UVのみの平均値フィルタ処理
def filter2d(src, m,n, fill_value=-1):
    # get kernel size
    # カーネルサイズ

    # width of skipｄｗ
    # 畳み込み演算をしない領域の幅
    d = int((m-1)/2)

    # get width height of input image
    # 入力画像の高さと幅
    h, w = src.shape[0], src.shape[1]

    # ndarray of destination
    # 出力画像用の配列
    if fill_value == -1:
        dst = src.copy()
    elif fill_value == 0:
        dst = np.zeros((h, w))
    else:
        dst = np.zeros((h, w))
        dst.fill(fill_value)

    # Spatial filtering
    # 畳み込み演算
    #論文では縦横をそれぞれ行っていたので分けてみた。
    for y in range(d, h - d):
        for x in range(d, w - d):
            dst[y][x][1] = (int)(np.sum(src[y-d:y+d+1, x,1] * (1/m) + 0.5))#U
            dst[y][x][2] = (int)(np.sum(src[y-d:y+d+1, x,2] * (1/m) + 0.5))#V
    for y in range(d, h - d):        
        for x in range(d, w - d):
            dst[y][x][1] = (int)(np.sum(src[y, x-d:x+d+1,1] * (1/m) + 0.5))#U
            dst[y][x][2] = (int)(np.sum(src[y, x-d:x+d+1,2] * (1/m) + 0.5))#V

    return dst

#ヒストグラム出力処理
def histgramer(hist):
    histgram = list()
    #plt.xlim(5, 55)                 # (1) x軸の表示範囲
    #plt.ylim(0, 30)                 # (2) y軸の表示範囲
    #plt.title("Store Visitors", fontsize=20)  # (3) タイトル
    #plt.xlabel("Age", fontsize=20)            # (4) x軸ラベル
    #plt.ylabel("Frequency", fontsize=20)      # (5) y軸ラベル
    plt.grid(True)                            # (6) 目盛線の表示
    #plt.tick_params(labelsize = 12)    # (7) 目盛線のラベルサイズ 

    for i in range(100):
        for j in range(hist[i]):
            histgram.append(i)
 
    # グラフの描画
    plt.hist(histgram , alpha=0.5,bins=50,histtype="step", color= 'c') #(8) ヒストグラムの描画
    print(hist)
    plt.show()

#画像取り出し
img = cv2.imread('./img/testI.jpg')
#print(img.shape)
#print(img)

#比較用や元データ取り出しなど。
img_origin = cv2.imread('./img/testI.jpg')
#画像ファイルの形調べ
#print(rgb2y(255,255,255),rgb2u(255,255,255),rgb2v(255,255,255))

#変数処理
color_rate = 0#色の比
data={0:0}
peak_lct = 0 #hist_peakの位置
peak_value = 0#hist_peakの値
pre_hist = 0 #一つ前のhist値
btm_lct = 0#histの谷
th = 0 #２値化のしきい値
hist = [0] * 100  #色のヒストグラム格納
pre_hist = [0] * 100 #処理前ヒストグラム
ave_hist = [0] * 100 #処理後ヒストグラム



"""
#CV2を使ったYUV変換及びYCrCb変換
#BGR→YUV
yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)#BGRからYUV
cv2.imwrite('./img/yuvtestI.jpg', yuv)

img_return = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)#YUVからBGR
cv2.imwrite('./img/returntestI.jpg', img_return)

ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)#YUVからYCrCb
cv2.imwrite('./img/ycrcbtestI.jpg', ycrcb)

"""
#画像を平均値フィルタ処理
img = cv2.blur(img, ksize=(3, 3))
img = cv2.blur(img, ksize=(3, 3))

#縦、横
#print(img.shape[0],img.shape[1])

#YUV変換処理
for i in range(img.shape[0]):#縦
    for j in range(img.shape[1]):#横
        for k in range(3):#画素
            if(k == 0):#Y
                img[i,j,k] = rgb2y(img[i,j,2],img[i,j,1],img[i,j,0])
            if(k == 1):#Cr
                img[i,j,k] = rgb2u(img[i,j,2],img[i,j,1],img[i,j,0])
            if(k == 2):#Cb
                img[i,j,k] = rgb2v(img[i,j,2],img[i,j,1],img[i,j,0])
                
#print(img) #中身確認

img_filter = filter2d(img, 5,5, -1)#UVのみ平均値フィルタ

cv2.imwrite('./img/filter_yuvtestI.jpg', img)#UV平均値フィルタ後の画像出力

#色差計算処理及びヒストグラムデータ作成
for i in range(img.shape[0]):#縦
    for j in range(img.shape[1]):#横
        #print(img[i,j,1]) #確認
        #print(img[i,j,2]) #確認
        color_rate = (int)(((img[i,j,2]/img[i,j,1]) * 100 - 50) + 0.5) #色差計算
        #print(color_rate)
        if (color_rate >= 0 and color_rate < 100):#1～100ならデータのカウント回数を増やす
            hist[color_rate] += 1
#print(hist)

#ヒストグラムデータ平均値フィルタ処理前の格納
for i in range(0,100):
    pre_hist[i] = hist[i]
    ave_hist[i] = hist[i]
histgramer(pre_hist)
    
#ヒストグラムを平滑化
for C in range(0,10):
    for i in range(0,100):
        temp_sum = 0
        for j in range(-2,3):
            m = i+j
            if(m < 0):
                m = 0
            if(m > 99):
                m = 99
            temp_sum = temp_sum + ave_hist[m]
        ave_hist[i] = (int)(temp_sum / 5.0 + 0.5)#0.5は四捨五入
    #print(ave_hist[i])
    #hist[i] = (int)(0.2*hist[i-2] + 0.2*hist[i-1] + 0.2*hist[i] + 0.2*hist[i+1] + 0.2*hist[i+2] + 0.5) #3個のデータでの平滑化
    histgramer(ave_hist)


'''
#平均値フィルタ複数回やるために実装
for i in range(1,99):
    ave_hist[i] = (int)(0.3*ave_hist[i-1] + 0.4*ave_hist[i] + 0.3*ave_hist[i+1] + 0.5)
    #hist[i] = (int)(0.2*hist[i-2] + 0.2*hist[i-1] + 0.2*hist[i] + 0.2*hist[i+1] + 0.2*hist[i+2] + 0.5)
histgramer(ave_hist)

for i in range(1,99):
    ave_hist[i] = (int)(0.3*ave_hist[i-1] + 0.4*ave_hist[i] + 0.3*ave_hist[i+1] + 0.5)
    #hist[i] = (int)(0.2*hist[i-2] + 0.2*hist[i-1] + 0.2*hist[i] + 0.2*hist[i+1] + 0.2*hist[i+2] + 0.5)
histgramer(ave_hist)

for i in range(1,99):
    ave_hist[i] = (int)(0.3*ave_hist[i-1] + 0.4*ave_hist[i] + 0.3*ave_hist[i+1] + 0.5)
    #hist[i] = (int)(0.2*hist[i-2] + 0.2*hist[i-1] + 0.2*hist[i] + 0.2*hist[i+1] + 0.2*hist[i+2] + 0.5)
histgramer(ave_hist)
'''

#ヒストグラムの編集を行うためデータを別の配列(辞書型)に入れる。
for i in range(0,100):
    data[i] = ave_hist[i]
    
#データのソート処理
data_sort = sorted(data.items(), key=lambda x:x[1], reverse=True)
#print(data_sort[0][0])
            
#ソート後に一番データが高い部分の番号とデータ変数を渡す。
peak_lct = data_sort[0][0]
peak_value = data_sort[0][1]
#print(peak_lct)
#print(peak_value)

#peakから色が赤くない方へ戻っていき、谷を見つける
pre_hist = peak_value
#print(pre_hist)
while(peak_lct >= 0):
    if(pre_hist - hist[peak_lct] >=0 or peak_value / 2 <= hist[peak_lct]):#現在のデータと前のデータを比べた判別式
        pre_hist = hist[peak_lct]
        #print(pre_hist)
        #if(pre_hist - hist[peak_lct] >=0)
    else:
        btm_lct = peak_lct
        print("find")
        print(btm_lct)
        break
    peak_lct -= 1
    #print(peak_lct)
    #print(hist[peak_lct])
    #print(btm_lct)

th = (btm_lct - 50) + 100 #しきい値決定
print(th)

#2値化処理
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if((img[i,j,2]*100)/(img[i,j,1] + 1) >= th):#2値化判別式
            img[i,j] = 255
        else:
            img[i,j] = 0
            
#出力画像
cv2.imwrite('./img/filter_yuvtestIbin.jpg', img)

