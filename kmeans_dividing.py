# coding: utf-8

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from sklearn.cluster import KMeans

EXPORT_DIR = "export/"
K = 4
N = 32

# devide img
IMG_FILE = "imgs/total_img_74.jpg"
img = cv2.imread(IMG_FILE)
h,w = img.shape[:2]
# print("h:",h,", w:",w)
y0 = int(h/N)
x0 = int(h/N)
s = min(y0,x0)*N
img = cv2.resize(img, (s,s), cv2.INTER_CUBIC)
h, w = img.shape[:2]
imgs = [img[x0*x:x0*(x+1), y0*y:y0*(y+1)] for x in range(N) for y in range(N)]
features = np.array([cv2.cvtColor(cv2.resize(i, (x0, y0), cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB) for i in imgs])
# print(features.shape)

train_data = features.reshape(features.shape[0], features.shape[1]*features.shape[2]*features.shape[3]).astype('float32') / 255.0
# print(train_data.shape)

# モデル定義
model = KMeans(n_clusters=K, init='k-means++', max_iter=5000, random_state=0)
y_res = model.fit_predict(train_data)

result_dict = dict(zip(range(0,len(y_res)),y_res))
result_list = sorted(result_dict.items(), key=lambda x:x[1])

# 結果出力
col = 10
row = int(len(features) / col) + 1
cols = 64*2
rows = 64*2
dpis = 150
fig = plt.figure(figsize=(cols, rows), dpi=dpis)
index = 1
for (i, label) in result_list:
    p = features[i]
    plt.subplot(row, col, index)
    plt.imshow(p, cmap='gray')
    plt.xlabel("cluster={}".format(label), fontsize=30)
    index += 1
    # print("index:"+str(index)+"\tlabel:"+str(label))
fig.savefig(EXPORT_DIR+'kmeans_divideing{}_result.jpg'.format(K))

colors=['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
W = w//N
H = h//N
fig2 = plt.figure(figsize=(N*4, N*4), dpi=dpis)
ax = plt.axes()
img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.imshow(img)
for i, label in result_dict.items():
    x,y = W*(i%N),H*(i//N)
    rect = patches.Rectangle(xy=(x,y),width=W, height=H, fc=colors[label],fill=True,alpha=0.3)
    ax.add_patch(rect)
    ax.text(x,y+W, str(label), size=W)
ax.figure.savefig(EXPORT_DIR+"kemans_dividing{}_result2.jpg".format(K))