import cv2
import numpy as np
import urllib.request

tail_width = 256
tail_height = 256

fmt_url = "https://saigai.gsi.go.jp/1/H30_07gouu/takahashigawa/photo/qv/{}-qv.jpg"
save_dir = "imgs/"
start = 0
end = 200

for i in range(start, end):
    image_url = fmt_url.format(str(i).zfill(4))
    print("req to "+image_url)
    
    req = urllib.request.Request(image_url)
    try:
        res = urllib.request.urlopen(req)
    except urllib.error.URLError as e:
        print(e.reason)
    else:
        body = res.read()
        img_buf = np.frombuffer(body, dtype=np.uint8)
        img = cv2.imdecode(img_buf, cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        cv2.imwrite(save_dir+"total_img_{}.jpg".format(i), img)
        print("save {}th file".format(i))