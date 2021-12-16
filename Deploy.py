import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('model_sign_language_classification.hdf5')
background = None
accumulated_weight = 0.5

ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350

def cal_accum_avg(frame, accumulated_weight):
    global background
    if background is None:
        background = frame.copy().astype('float')
        return None
    cv2.accumulateWeighted(frame, background, accumulated_weight)

def segment_hand(frame, threshold = 25): #  Hàm phân cùng giữa tay và nền
    global background
    diff = cv2.absdiff(background.astype('uint8'), frame) #  Tính toán sự khác biệt tuyệt đối cho mỗi phần tử giữa hai
                                                            # mảng hoặc giữa một mảng và một đại lượng vô hướng.
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    #   Lấy các đường viền bên ngoài cho hình ảnh
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #  findContours(image, mode, method[, contours[, hierarchy[, offset]]])
    if len(contours) == 0:
        return None
    else:
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        return (thresholded, hand_segment_max_cont)
word_dict = {0:'one', 1:'two', 2:'three', 3:'four', 4:'five'}
cam = cv2.VideoCapture(0)
num_frames = 0
while True:
    ret, frame = cam.read()
    # Lật khung ảnh để không bị lật ảnh khi chụp hình
    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()
    roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]
    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (9,9), 0) # Lọc hình ảnh là cho min, mượt

    if num_frames < 60:
        cal_accum_avg(gray_frame, accumulated_weight)
        cv2.putText(frame_copy, 'FETCHING BACKGROUND...PLEASE WAIT!', (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0,0,255), 2)
            #   putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]) -> img
    else:
        hand = segment_hand(gray_frame)
        if hand is not None:
            thresholded, hand_segment = hand
            cv2.drawContours(frame_copy, [hand_segment+(ROI_right, ROI_top)], -1, (255,0,0), 1)
            cv2.imshow('Threshold image', thresholded)
            thresholded = cv2.resize(thresholded, (150,150))
            thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
            thresholded = np.reshape(thresholded, (1,thresholded.shape[0], thresholded.shape[1], 3))
            pred = model.predict(thresholded)
            cv2.putText(frame_copy, word_dict[np.argmax(pred)], (170,45), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 2)

    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255, 128, 0), 3)
    num_frames += 1
    cv2.putText(frame_copy, 'Hand sign recognition - Sol___', (10,20), cv2.FONT_ITALIC, 0.5,
                (51, 255, 51), 1)
    cv2.imshow('Sign detection', frame_copy)
    # ĐÓng cửa sổ với x
    k = cv2.waitKey(1) & 0xFF
    if k == ord('x'):
        break

#   Tắt máy ảnh và đóng các cửa sổ
cam.release()
cv2.destroyAllWindows()
