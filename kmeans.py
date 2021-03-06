# coding: utf-8
 
from PIL import Image
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

IMG_DIR = "imgs/"
EXPORT_DIR = "export/"
K = 3

files = glob.glob(IMG_DIR+"*")

col = 10
row = int(len(files) / col) + 1
cols = 64
rows = 64
dpis = 100

fig = plt.figure(figsize=(cols, rows), dpi=dpis)

for f in files:
    try:
        img = Image.open(f)
        # print("open "+f)
        img_resize = img.resize((64, 64))
    except OSError as e:
        print(e)
        quit()
        

features = np.array([cv2.cvtColor(cv2.resize(cv2.imread(f), (64, 64), cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB) for f in files])
print(features.shape)

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

fig.savefig(EXPORT_DIR+'kmeans_result.jpg')