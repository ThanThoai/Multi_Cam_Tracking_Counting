# from imutils.video import VideoStream
import imagezmq
import cv2
import urllib.request
import numpy as np


# path = "rtsp://192.168.1.101:8080/h264_ulaw.sdp"  # change to your IP stream address
# path = 'https://192.168.100.137:6969/video'
# cap = cv2.VideoCapture(path)

# # change to IP address and port of server thread
# sender = imagezmq.ImageSender(connect_to='tcp://localhost:5555')
# cam_id = 'Camera 1'  # this name will be displayed on the corresponding camera stream
# print('aaaa')
# print('-------------------------------------')
# # stream = cap.start()


cap = cv2.VideoCapture()
# Opening the link
cap.open("http://192.168.1.101:8080/video?.mjpeg")
sender = imagezmq.ImageSender(connect_to='tcp://localhost:5555')
cam_id = 'Camera 1'
# print("INFO")

while True:
    print("camera:", cam_id)
    _, frame = cap.read()
    sender.send_image(cam_id, frame)
