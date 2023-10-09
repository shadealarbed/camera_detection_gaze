import dlib
import cv2
import numpy as np
from image_resize import ImageProcessor
import psutil
process = psutil.Process()
import os


import tkinter as tk
from tkinter import filedialog

# Create a tkinter root window (it won't be shown)
root = tk.Tk()
root.withdraw()  # Hide the root window
model_path = ""

# Define a file type filter for JPG files
filetypes = [("JPEG files", "*.jpg")]
# Open a file dialog to pick a file
image_path = filedialog.askopenfilename(
    title="Select a JPG File",
    filetypes=filetypes
)

# Initialize dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
# Define the relative path to the directory containing the shape predictor file
relative_directory = ""

# Get the absolute path of the current working directory
current_directory = os.getcwd()

# Construct the absolute directory path
absolute_directory = os.path.abspath(os.path.join(current_directory, relative_directory))

# Define the name of the shape predictor file
file_name = "/Users/shadialarbed/Desktop/image_recognition/landmarks_model/facial_landmark_model.dat"

# Construct the full file path
predictor_path = os.path.join(absolute_directory, file_name)

# Load the shape predictor
predictor = dlib.shape_predictor(predictor_path)
processor = ImageProcessor(image_path)

frame = cv2.imread(image_path)



# Convert the image to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


faces = detector(gray)

## constants
margin = 100
light = False
depth = False
light_msg = ""
depth_message = ""

def get_landmarks(gray, face):
    landmarks1 = predictor(gray, face)
    return landmarks1, np.array([[point.x, point.y] for point in landmarks1.parts()])

def roi_marks(landmarks,frame):
    min_x1, max_x1, min_y1, max_y1 = mapXY(landmarks)
    return frame[int(min_y1):int(max_y1), int(min_x1):int(max_x1)]

def color_threshold(threshold_value,roi_face_light):
    if len(roi_face_light) !=0:
        gray1 = cv2.cvtColor(roi_face_light, cv2.COLOR_BGR2GRAY)
    
        return cv2.threshold(gray1, threshold_value, 255, cv2.THRESH_BINARY)
    else:
        print("no faces to detected")

def detect_roi_face_light(frame, landmarks):
    roi_face_light = roi_marks(landmarks,frame)
    if len(roi_face_light) != 0:
        _, thresholded_face = color_threshold(250,roi_face_light)
        _, thresholded_face_low = color_threshold(10,roi_face_light)
    else:
        print("no faces")
    return thresholded_face, thresholded_face_low


def calculate_depth(landmarks):
    """
    Calculate the depth or distance to the camera based on facial landmarks.

    Args:
        landmarks (np.ndarray): Array of facial landmarks.

    Returns:
        float: The calculated depth or distance to the camera.
    """
    # Calculate the standard deviation of the x-coordinates of landmarks
    return np.std(landmarks[:, 0])
 
def measure_depth(landmarks):

    std_dev = calculate_depth(landmarks)
    if std_dev <= 60:
        depth_message = "too far"
        depth = False
    elif std_dev >= 130:
        depth_message = "too close"
        depth = False
    else:
        depth_message = "good"
        depth = True
        
    return f"Distance to camera is {depth_message}: {std_dev:.2f}",depth

def measure_brightness(frame, landmarks):
    thresholded_face, thresholded_face_low = detect_roi_face_light(frame, landmarks)
    bright_pixel_count = np.sum(thresholded_face == 255)
    bright_pixel_count_low = np.sum(thresholded_face_low == 255)
    if bright_pixel_count >= 3000:
        light_msg =  f"too high brightness {bright_pixel_count}"
        light = False
    # elif bright_pixel_count_low >= 2000:
    #     message = "too low brightness"
    else:
        light_msg =  "good brightness"
        light = True
        
    return light_msg , light
  
############ claculate face diraction
def get_side_landmarks(left_landmarks,right_landmarks):
    return np.std(left_landmarks[:, 0]) if len(left_landmarks) > 0 else 0,np.std(right_landmarks[:, 0]) if len(right_landmarks) > 0 else 0

def get_vertical_distance(landmarks1):
    # Define the coordinates of landmarks 1, 15, and 29
    x1, y1 = landmarks1.part(1).x, landmarks1.part(1).y
    x15, y15 = landmarks1.part(15).x, landmarks1.part(15).y
    x29, y29 = landmarks1.part(29).x, landmarks1.part(29).y
    
    # Calculate the vertical distance (VD) from point 29 to the line connecting points 1 and 15
    return np.array([x1, y1]),np.array([x15, y15]),np.array([x29, y29])

def claculate_distence_between_landmarks(landmarks1):
    # Calculate the vertical distance (VD) from point 29 to the line connecting points 1 and 15
    p1 ,p2 ,p3 = get_vertical_distance(landmarks1)
    # Calculate the Euclidean distances between landmarks 1 and 29, and between landmarks 15 and 29
    return np.linalg.norm(p1 - p3),np.linalg.norm(p2 - p3)

def clac_theta_between_landmarks(p1,p2,p3,distance_1_29, distance_15_29 ):
    VD = np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1)
    return np.arcsin(VD / distance_1_29),np.arcsin(VD / distance_15_29)

def compined_dis_points(landmarks1):
    p1 ,p2 ,p3 = get_vertical_distance(landmarks1)

    
    distance_1_29, distance_15_29 = claculate_distence_between_landmarks(landmarks1)
    
    return clac_theta_between_landmarks(p1,p2,p3,distance_1_29, distance_15_29)

def pitched_angle(landmarks1):
    theta_1 , theta_2 =  compined_dis_points(landmarks1)
    
    return (theta_1 + theta_2) / 2


def left_right_side(left_landmarks, right_landmarks):
    left_std,right_std = get_side_landmarks(left_landmarks,right_landmarks)
    
    if left_std - right_std > 20:
        return "Left"
    elif right_std - left_std > 25:
        return "Right"
    

def up_down_diractions(landmarks1):
    pitch_angle = pitched_angle(landmarks1)

    if pitch_angle > 0.15:
        return "Down"
    elif pitch_angle < - 0.15:
        return "Up"
    

############ calculate face diraction

##
def estimate_face_direction(left_landmarks, right_landmarks, landmarks1):
    if left_right_side(left_landmarks, right_landmarks) or up_down_diractions(landmarks1):
        return left_right_side(left_landmarks, right_landmarks) or up_down_diractions(landmarks1)
    else:
        return "Center"
##

def detect_direction(landmarks, landmarks1):
    center_x = np.mean(landmarks[58:64, 0])
    return estimate_face_direction(
        left_landmarks=landmarks[landmarks[:, 0] < center_x],
        right_landmarks=landmarks[landmarks[:, 0] >= center_x],
        landmarks1=landmarks1
    )
################### capture face area conditions #############################
def mapXY(landmarks):
    return np.min(landmarks[:, 0]), np.max(landmarks[:, 0]), np.min(landmarks[:, 1]), np.max(landmarks[:, 1])

def margen_cordents(landmarks):
    min_x, max_x, min_y, max_y = mapXY(landmarks)
    min_x -= margin
    min_y -= margin + 50
    max_x += margin
    max_y += margin - 50
    return min_x ,min_y,max_x,max_y


def detect_margen_cordents(landmarks,frame):
    min_x ,min_y,max_x,max_y = margen_cordents(landmarks)
    return max(min_x, 0),max(min_y, 0),min(max_x, frame.shape[1]),min(max_y, frame.shape[0])


def detect_roi_face(frame, landmarks):
    min_x ,min_y,max_x,max_y = detect_margen_cordents(landmarks,frame)
    return frame[int(min_y):int(max_y), int(min_x):int(max_x)]
################### capture face area conditions #############################

def calling_cond(shape,shape1,frame):
    detect_roi_face_light(frame, shape)
    light_msg,light = measure_brightness(frame, shape)
    
    direction = detect_direction(shape, shape1)
    return detect_roi_face_light(frame, shape),light_msg,light,direction
def get_landmarks(gray,faces):
    return predictor(gray, faces[0]),np.array([(predictor(gray, faces[0]).part(i).x, predictor(gray, faces[0]).part(i).y) for i in range(68)])

def draw_landmarks(frame,landmarks):
    for (x, y) in landmarks:
        cv2.circle(frame, (x, y), 2, (0,0,255), -1)


if len(faces) == 1: 
    # Get the facial landmarks for the detected face
    shape1,shape = get_landmarks(gray,faces)
    roi_face = detect_roi_face(frame, shape)
    
    roi ,light_msg,light,direction = calling_cond(shape,shape1,frame)
    

    detect_roi_face_light(frame, shape)
    light_msg,light = measure_brightness(frame, shape)
    
    direction = detect_direction(shape, shape1)
    draw_landmarks(frame,shape)


    if light == True:
        cv2.imwrite("/Users/shadialarbed/Desktop/image_recognition/optimized_face_detection/images/image_filtered.jpg",roi_face)
    elif light == False:
        print(light_msg)
    # elif direction != "Center":
    #     print(f"the direction is : {direction} it should be Center")
    else:
        print("something is wrong give another pick")

elif len(faces) > 1:
    print("there is to many faces in the picture")
    print(f"number of faces is : {len(faces)}")
else:
    print("error no faces")


print(f"memory used: {process.memory_info().rss}")