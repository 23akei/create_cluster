# coding: utf-8

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

EXPORT_DIR = "export/"
K = 3
N = 16

# devide img
IMG_FILE = "imgs/total_img_74.jpg"
img = cv2.imread(IMG_FILE)
h,w = img.shape[:2]
y0 = int(h/N)
x0 = int(h/N)
imgs = [img[x0*x:x0*(x+1), y0*y:y0*(y+1)] for x in range(N) for y in range(N)]
features = np.array([cv2.cvtColor(cv2.resize(i, (x0, y0), cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB) for i in imgs if i.size != 0])
print(features.shape)

col = 10
row = int(len(features) / col) + 1
cols = 64*2
rows = 64*2
dpis = 150

train_data = features.reshape(features.shape[0], features.shape[1]*features.shape[2]*features.shape[3]).astype('float32') / 255.0
print(train_data.shape)

# モデル定義
model = KMeans(n_clusters=K, init='k-means++', max_iter=5000, random_state=0)
y_res = model.fit_predict(train_data)

result_dict = dict(zip(range(0,len(y_res)),y_res))
result_list = sorted(result_dict.items(), key=lambda x:x[1])
# 結果出力

fig = plt.figure(figsize=(cols, rows), dpi=dpis)
index = 1

for (i, label) in result_list:
    p = features[i]
    plt.subplot(row, col, index)
    plt.imshow(p, cmap='gray')
    plt.xlabel("cluster={}".format(label), fontsize=30)
    index += 1
    # print("index:"+str(index)+"\tlabel:"+str(label))

fig.savefig(EXPORT_DIR+'kmeans_devideing_result.jpg')