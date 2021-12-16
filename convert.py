import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
import ipywidgets as widgets
import io
from IPython.display import display, clear_output
import ipywidgets as widgets
import warnings

model = load_model('sign_language_classification_2.h5')
img = cv2.imread('test5_5.png')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.GaussianBlur(img_gray, (9,9), 0)
img = cv2.resize(img_gray, (150,150))
img = img.reshape(1, 150, 150, 3)
ret, thresholded = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
p = model.predict(thresholded)
p = np.argmax(p,axis=1)[0]
if p==0:
    p='one'
elif p==1:
    p='two'
elif p==2:
    p = 'three'
elif p==3:
    p='four'
elif p == 4:
    p='five'
else:
    print('Can not dectect your hand!')

print(p)


# from IPython.display import display, clear_output
# import ipywidgets as widgets
# import io
# uploader = widgets.FileUpload()
# display(uploader)
# button = widgets.Button(description='Predict')
# out = widgets.Output()
# def on_button_clicked(_):
#     with out:
#         clear_output()
#         try:
#             img_pred(uploader)
#         except:
#             print('No Image Uploaded/Invalid Image File')
# button.on_click(on_button_clicked)
# widgets.VBox([button,out])
# img = cv2.imread('test1.jpg')
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_gray = cv2.GaussianBlur(img_gray, (9,9), 0)
# ret, thresholded = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
