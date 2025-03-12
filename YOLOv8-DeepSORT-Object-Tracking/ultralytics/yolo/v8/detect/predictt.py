import ultralytics
import hydra
import torch
import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from ultralytics import YOLO 
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

import cv2
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import numpy as np
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

# Add these global variables to track car counts for both lines
count_line_1 = 0
count_line_2 = 0
deepsort = None
data_deque = {}  # To track object movement across frames

def draw_two_lines(img, line_1, line_2):
    """
    Function to draw two lines on the image.
    """
    # Extend the length of the lines by increasing the x-coordinates
    line_1_extended = [(50, line_1[0][1]), (750, line_1[1][1])]  # Green Line near the signal (longer)
    line_2_extended = [(50, line_2[0][1]), (750, line_2[1][1])]  # Red Line far from the signal (longer)

    # Increase the distance between the two lines by adjusting the y-coordinates
    new_line_1_y = line_1[0][1] + 100  # Move the first line down by 100 pixels
    new_line_2_y = line_2[0][1] - 100  # Move the second line up by 100 pixels
    
    # Update the lines with new y-coordinates
    line_1_moved = [(50, new_line_1_y), (750, new_line_1_y)]
    line_2_moved = [(50, new_line_2_y), (750, new_line_2_y)]

    # Draw the lines on the image
    cv2.line(img, line_1_moved[0], line_1_moved[1], (0, 255, 0), 3)  # Green Line (near the signal)
    cv2.line(img, line_2_moved[0], line_2_moved[1], (0, 0, 255), 3)  # Red Line (far from the signal)

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
    
    return img

def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def check_line_crossing(center, line_y_pos):
    """
    Function to check if the car's center has crossed a specific line.
    Returns True if crossed.
    """
    return center[1] > line_y_pos  # Line crossing condition based on the y-position

def UI_box(x, img, label=None, color=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def draw_boxes(img, bbox, names, object_id, identities=None, offset=(0, 0), line_1_y=None, line_2_y=None):
    global count_line_1, count_line_2

    height, width, _ = img.shape

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # Code to find center of bottom edge
        center = (int((x2 + x1) / 2), int((y2 + y2) / 2))

        # Get ID of object
        id = int(identities[i]) if identities is not None else 0

        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label = '{}{:d}'.format("", id) + ":" + '%s' % (obj_name)

        UI_box(box, img, label=label, color=color, line_thickness=2)

        # Adding condition to count cars only moving south (crossing line 1 after line 2)
        car_passed_line_2 = False  # Track if a car crossed line 2
        
        # Check if the car crosses line 2 (far from signal)
        if check_line_crossing(center, line_2_y):
            count_line_2 += 1
            car_passed_line_2 = True
            data_deque[id] = None  # Reset deque for the car after counting
        
        # Check if the car crosses line 1 (near signal) only after it has crossed line 2
        if car_passed_line_2 and check_line_crossing(center, line_1_y):
            count_line_1 += 1
            data_deque[id] = None  # Reset deque after counting

        # Draw the trail for the car
        if id in data_deque:
            for i in range(1, len(data_deque[id])):
                if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                    continue
                thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
                cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)

    # Display the counts and the difference
    cv2.putText(img, f'Line 1 Count: {count_line_1}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f'Line 2 Count: {count_line_2}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Calculate the difference between the two counts (number of cars in the south direction between the lines)
    difference = count_line_1 - count_line_2
    cv2.putText(img, f'Difference: {difference}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def process_video(video_path):
    """
    Function to process the video and apply car counting logic.
    """
    cap = cv2.VideoCapture(video_path)

    # Define positions of the lines (y-coordinates)
    line_1_y = 400  # Adjust this value based on your video
    line_2_y = 300  # Adjust this value based on your video
    
    # Coordinates for drawing lines (change these as per your video's resolution)
    line_1 = [(100, line_1_y), (600, line_1_y)]  # Green line near the signal
    line_2 = [(100, line_2_y), (600, line_2_y)]  # Red line far from the signal

    # Loop through the video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process each frame: here you would integrate the object detection to get 'bbox' and 'identities'
        bbox = []  # Bounding box for each detected vehicle (replace with actual detection logic)
        identities = []  # IDs for each object (replace with actual detection logic)
        object_id = []  # Object class ID (replace with actual detection logic)
        names = ['car']  # Name list for object classes (replace with actual classes)
        
        # Draw the two lines on the frame
        draw_two_lines(frame, line_1, line_2)
        
        # Call draw_boxes to check line crossing and display counts
        draw_boxes(frame, bbox, names, object_id, identities, line_1_y=line_1_y, line_2_y=line_2_y)

        # Display the processed frame
        cv2.imshow('Traffic Management', frame)

        # Slow down the video playback for better visualization
        if cv2.waitKey(30) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# Call the process_video function with your video path
process_video('test3.mp4')