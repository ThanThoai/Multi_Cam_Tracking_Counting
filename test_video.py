import imagezmq
import cv2
import numpy as np



sender = imagezmq.ImageSender(connect_to='tcp://localhost:5555')

cam_id = 'Camera 1'
path_video = 'video_test/00000000204000400.mp4'
cap = cv2.VideoCapture(path_video)
frames = 0

while cap.isOpened():
    ret, frame  = cap.read()
    # frames += 1
    # if frames % 20 == 0:
    #     if not ret:
    #         break 
    sender.send_image(cam_id, frame)
