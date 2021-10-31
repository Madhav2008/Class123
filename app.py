import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

if(not os.environ.get("PYTHONHTTPSVERIFY", "")and getattr(ssl, "_create_unverified_context", None)):
    ssl._create_default_https_context = ssl._create_unverified_context

X, y = fetch_openml("mnist_784", version = 1, return_X_y = True)
print(pd.Series(y).value_counts())

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
nclasses = len(classes)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)
X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

clf = LogisticRegression(solver = "saga", multi_class = "multinomial").fit(X_train_scaled, Y_train)

Y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy Is : " + str(accuracy))

cap = cv2.VideoCapture(0)

while(True):
    try:
        ret, frame = cap.read()
        gray = cv2.cvtcolor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        upperleft = (int(width / 2 - 56), int(width / 2 - 56))
        bottomright = (int(width / 2 + 56), int(width / 2 + 56))
        cv2.rect(gray, upperleft, bottomright, (0, 255, 0), 2)
        roi = gray[upperleft[1]:bottomright[1], upperleft[0]:bottomright[0]]

        pil = Image.fromarray(roi)
        img_bw = PIL.convert("L")
        img_bw_resized = img_bw.resize((28, 28), Image.ANTIALIAS)
        img_bw_resized_inverted = PIL.ImageOps.invert(img_bw_resized)
        pixel_filter = 20
        min_pixel = np.percentile(img_bw_resized_inverted, pixel_filter)
        img_bw_resized_inverted_scaled = np.clip(img_bw_resized_inverted - min_pixel, 0, 255)
        max_pixel = np.max(img_bw_resized_inverted)
        img_bw_resized_inverted_scaled = np.asarray(img_bw_resized_inverted_scaled) / max_pixel
        test_sample = np.array(img_bw_resized_inverted_scaled).reshape(1, 784)
        test_pred = clf.predict(test_sample)
        print(test_pred)
        cv2.imshow("frame", gray)
        if cv2.waitKey(1)& 0xFF == ord("q"):
            break
    
    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()