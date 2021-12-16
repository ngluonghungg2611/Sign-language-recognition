import warnings
import numpy as np
import cv2


warnings.simplefilter(action='ignore', category=FutureWarning)

background = None
accumulated_weight = 0.5 # tích lữu trọng số
#   Tạo kích thước cho cho cửa sổ làm việc
ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350

def cal_accum_avg(frame, accumulated_weight): # Hàm cân chỉnh nền
    global background #  thay đổi giá trị của biến toàn cục trong một hàm
    if background is None:
        background = frame.copy().astype('float')
        return None
    cv2.accumulateWeighted(frame, background, accumulated_weight)
'''
accumulateWeighted(src, dst, alpha[, mask]):
    - Hàm tính tổng trọng số của hình ảnh đầu vào src và tích lũy vào dst để dst trở thành giá trị trung bình chạy của
      mỗi khung hình
'''

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


#   Kết nối camera để lấy hình
cam = cv2.VideoCapture(0)
num_frames = 0
element = 1
num_imgs_taken = 0
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
        if num_frames <= 59:
            cv2.putText(frame_copy, 'FETCHING BACKGROUND...PLEASE WAIT!', (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0,0,255), 2)
            #   putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]) -> img

    #   Thời gian để định cấu hình tay cụ thể trong ROI
    elif num_frames <= 300:
        hand = segment_hand(gray_frame)
        cv2.putText(frame_copy, 'Adjust hand... gesture for' + str(element), (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

        #   Kiểm tra xem bàn tay có thực sự được detect ngay hay không bằng cách đếm số
        if hand is not None:
            thresholded, hand_segment = hand
            #   Vẽ contours xung quanh hand đã được segment
            cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0,0), 1)
            cv2.putText(frame_copy, str(num_frames) + 'for' + str(element), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0,0,255), 2)

            # Cùng lúc đó hiển thị threshold image
            cv2.imshow('Threshold Hand image', thresholded)
    else:
        #   Segment bàn tay
        hand = segment_hand(gray_frame)
        #   Kiểm tra xem có detect được hand không
        if hand is not None:
            #   lấy thresholded and hand_segment_max_cont
            thresholded, hand_segment = hand
            #   Vẽ contours xung quanh hand đã được segment
            cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0), 1)
            cv2.putText(frame_copy, str(num_frames), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.putText(frame, str(num_imgs_taken) + 'image' + 'for' + str(element), (200, 400), cv2.FONT_HERSHEY_SIMPLEX,
                         1, (0,0,255), 2)

            #   Hiển thị threshold image
            cv2.imshow('Threshold hand image', thresholded)
            if num_imgs_taken <= 300:
                cv2.imwrite(r'D:\\COmputerVision\\Sign_Languge_Recognition\\train\\' + str(element) + '\\' +
                            str(num_imgs_taken+300) + '.jpg', thresholded)
            else:
                break
            num_imgs_taken += 1
        else:
            cv2.putText(frame_copy, 'No hand detected...', (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)


    #   Vẽ ROI trên frame_copy
    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255, 128, 0), 3)
    cv2.putText(frame_copy, 'DataFlair hand sign recogotion___', (10, 30), cv2.FONT_ITALIC, 0.5, (51,255,51),1)

    #   Tăng số lượng khung hình để theo giõi
    num_frames += 1

    # Hiển thị frame với hand segment
    cv2.imshow('Sign Detection', frame_copy)

    # ĐÓng cửa sổ với x
    k = cv2.waitKey(1) & 0xFF
    if k == ord('x'):
        break

#   Tắt máy ảnh và đóng các cửa sổ
cv2.destroyAllWindows()
cam.release()











