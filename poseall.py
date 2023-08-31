from select import select
import cv2
#from matplotlib import offsetbox
#from matplotlib.text import OffsetFrom
import mediapipe as mp
import numpy as np
from numpy import linalg as LA, r_
from scipy.signal import savgol_filter
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt
import time
from datetime import datetime
import math

from pandas import DataFrame
import config

from select_region import *
from testingmath import *

font = cv2.FONT_HERSHEY_SIMPLEX

timestamp = time.time()
date_time = datetime.fromtimestamp(timestamp)
str_date_time = date_time.strftime("%d-%m-%Y, %H:%M:%S")

real_time = config.real_time
if(not config.real_time):
    video_name = config.video_name
if(config.real_time):
    video_name = time.time()

#신뢰값
pose_confidence_value = 0.9

meter_per_pixel =0
print("====== meters per pixel=========")
tee_stand_pos = (config.box_end[0]+config.box_start[0])/2 #teeball stand in pixels

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

offset_frames = 0

prev_elbow_angle = 0
prev_wrist_angle = 0

max_clockwise_body_turn = 0
max_clockwise_body_turn_frame = 0

highest_foot_pos = 1
backswing_start_frame = 0
swing_start_frame = 0
swing_end_frame = 0
prev_wrist_max_pos = 0
framenumber = []


right_hip_position = []
left_hip_position = []
right_shoulder_position = []
left_shoulder_position = []
right_wrist_position = []
left_wrist_position = []
right_elbow_position = []
left_elbow_position = []
right_knee_position = []
left_knee_position = []
right_ankle_position = []
left_ankle_position = []
right_foot_position = []
left_foot_position = []

l_wrist_angle_data = []
r_wrist_angle_data = []

body_turn_data = []
elbow_angle_data = []
hip_turn_data = []

r_ankle_angle_data = []
l_ankle_angle_data = []

r_elbow_angle_data = []
l_elbow_angle_data = []

r_knee_angle_data = []
l_knee_angle_data = []

r_should_sag_data = []
r_should_front_data = []
r_should_horiz_data = []

l_should_sag_data = []
l_should_front_data = []
l_should_horiz_data = []

r_hip_sag_data = []
l_hip_sag_data = []

r_hip_horiz_data = []
l_hip_horiz_data = []

l_elbow_horizontal_angle_data = []
r_elbow_horizontal_angle_data = []

l_ankle_frontal_angle_data = []
r_ankle_frontal_angle_data = []

l_hip_frontal_angle_data = []
r_hip_frontal_angle_data = []

l_wrist_frontal_angle_data = []
r_wrist_frontal_angle_data = []

ball_position = []
form_condition_heel_passed_box_start = False
heel_passed_box_start_coordinate = 0
toe_passed_box_end_coordinate = 0
form_condition_heel_passed_box_end = False
form_condition_foot = False
form_condition_backswing=False
form_condition_forwardswing=False
form_condition_sequence=False

backswing_fail_string = "백스윙 시 양쪽 엄지발가락 쪽에 힘을 모으고, 골반과 무릎을 굽혀 모여진 모습으로 백스윙 한 후 순간적으로 타격 하세요"
foot_fail_string = "뻗는 발을 접을 때 몸이 뒤로 기울지 않게 하고, 뻗는 위치는 티대까지만 뻗으세요"
forwardswing_fail_string = "몸을 비스듬히 하여 무릎과 골반을 사용하고, 뒤쪽 팔꿈치를 몸의 안쪽(가슴쪽)으로 가까이하여 스윙하세요"
sequence_fail_string = "골반을 먼저 회전하고,  팔꿈치를 안쪽으로 하여 임팩트 후 멈추지 말고 머리에 공이 있다고 생각하여 끝까지 회전하세요"
success_string = "골반-몸통-팔꿈치 순서로 움직임이 잘 나타났습니다"

#start = False
start = True
do_once = True

class Highlighter(object):
    def __init__(self, ax, x, y):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.x, self.y = x, y
        self.mask = np.zeros(x.shape, dtype=bool)

        self._highlight = ax.scatter([], [], s=200, color='yellow', zorder=10)

        self.selector = RectangleSelector(ax, self, useblit=True)

    def __call__(self, event1, event2):
        self.mask |= self.inside(event1, event2)
        xy = np.column_stack([self.x[self.mask], self.y[self.mask]])
        self._highlight.set_offsets(xy)
        self.canvas.draw()

    def inside(self, event1, event2):
        """Returns a boolean mask of the points inside the rectangle defined by
        event1 and event2."""
        # Note: Could use points_inside_poly, as well
        x0, x1 = sorted([event1.xdata, event2.xdata])
        y0, y1 = sorted([event1.ydata, event2.ydata])
        mask = ((self.x > x0) & (self.x < x1) &
                (self.y > y0) & (self.y < y1))
        return mask

class detection:
    def __init__(self, frame, x, y):
        self.frame = frame
        self.x = x
        self.y = y

def ball_tracking(ret, frame, frame_next, position, currentframe):
    if not ret:
        print("ball cannot be tracked")
        return
    diff = cv2.absdiff(frame, frame_next)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, movement_mask = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY) 

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowerhsv = np.array([0, 160, 80])
    upperhsv = np.array([50, 255, 255])

    color_mask = cv2.inRange(hsv, lowerhsv, upperhsv)\
    
    combined_mask = cv2.bitwise_and(movement_mask, color_mask)
    result = cv2.bitwise_and(frame, frame, mask = combined_mask)
    dilated = cv2.dilate(combined_mask, None, iterations = 3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 300:
             continue
        (x,y), radius=cv2.minEnclosingCircle(contour)
        center=(int(x), int(y))
        radius=int(radius)
        cv2.circle(result, center, radius, (0, 255, 0), 2)
        position.append(detection(currentframe, x, y))

    cv2.imshow("frame", frame)
    cv2.imshow("color mask", color_mask)
    cv2.imshow("movement mask", movement_mask)
    cv2.imshow("result", result)
    
    #currentframe += 1

def calculate_body_turn(shoulder_vec, hip_vec):#input must be 2d vectors
    v1=shoulder_vec/LA.norm(shoulder_vec)
    v2=hip_vec/LA.norm(hip_vec)
    res = np.dot(v1, v2)
    #print('dot prod for body is : ', res)
    angle_rad = np.arccos(res)
    ret_angle=0
    #print('body angle : ', math.degrees(angle_rad))
    if np.cross(v1, v2)<0:
        ret_angle = -1*math.degrees(angle_rad)
        return ret_angle
    else:
        ret_angle = math.degrees(angle_rad)
        return ret_angle
    #print(ret_angle)
    

def findangle(vec1, vec2):
    vec1_unit = vec1 / LA.norm(vec1)
    vec2_unit = vec2 / LA.norm(vec2)
    angle_rad = np.arccos(np.dot(vec1_unit, vec2_unit))
    return math.degrees(angle_rad)

def calculate_angle_2d(start, middle, end):
    v1 = np.array([start[0] - middle[0], start[1] - middle[1]])
    v2 = np.array([end[0] - middle[0], end[1] - middle[1]])
    v1mag = np.sqrt(v1[0]**2 + v1[1]**2)
    v2mag = np.sqrt(v2[0]**2 + v2[1]**2)
    v1norm = np.array([v1[0]/v1mag, v1[1]/v1mag])
    v2norm = np.array([v2[0]/v2mag, v2[1]/v2mag])
    res = np.dot(v1norm, v2norm)
    angle_rad = np.arccos(res)
    return math.degrees(angle_rad)

def calculate_angle_3d(start, middle, end):
    v1 = np.array([start[0] - middle[0], start[1] - middle[1], start[2]-middle[2]])
    v2 = np.array([end[0] - middle[0], end[1] - middle[1], end[2] - middle[2]])
    v1mag = LA.norm(v1)
    v2mag = LA.norm(v2)
    v1norm = np.array([v1[0]/v1mag, v1[1]/v1mag, v1[2]/v1mag])
    v2norm = np.array([v2[0]/v2mag, v2[1]/v2mag, v2[1]/v2mag])
    res = np.dot(v1norm, v2norm)
    if(res>1):
        res=1
    elif(res<-1):
        res=-1
    angle_rad = np.arccos(res)
    return math.degrees(angle_rad)
    

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence = pose_confidence_value, enable_segmentation = True) as pose:
    currentframe = 0
    if(real_time == 0): #if pre recorded
        cap = cv2.VideoCapture(video_name)
    else:
        cap = cv2.VideoCapture(0)
        if(cap.isOpened()): print("CAM OPEN")
    _, frame = cap.read()
    while(not _): 
        _, frame = cap.read()
    if(not _) : print("FRAME NOT READ")
    select_region(cap)
    meter_per_pixel = config.base_length/abs(config.box_end[0] - config.box_start[0])
    if(config.real_time):
        #cap = cv2.VideoCapture(0)
        while True:
            #ret, frame = cap.read()
            start_time = datetime.now()
            diff = (datetime.now() - start_time).seconds
            while(diff <= config.timer):
                ret, frame = cap.read()
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, "paused for {} seconds, currently {} seconds".format(config.timer, diff), (10,450), font, 1, (0,255,0), 2, cv2.LINE_AA)
                cv2.imshow("delay window, will resume after {} seconds".format(config.timer), frame)
                diff = (datetime.now() - start_time).seconds
            break
    

    while cap.isOpened():
        width  = cap.get(3)
        height = cap.get(4)
        ret, frame = cap.read()
        ret, frame_next = cap.read()
        
        

        if not ret:
            print("can't receive frame. exiting")
            break

        ball_tracking(ret, frame, frame_next, ball_position, currentframe)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #image = cv2.flip(image, 1)
        image.flags.writeable = False
        
        #extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            if do_once:
                #calculate global x axis once
                global_x_right = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]
                global_x_left = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
                global_x_vec = np.array([global_x_right.x - global_x_left.x, global_x_right.z - global_x_left.z])

                do_once=False
            if True:
                #print('*****************')
                #right left hip landmark

                
                rhip_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                lhip_landmark = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]

                #right left shoulder landmark
                rshoulder_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                lshoulder_landmark = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                #print(rshoulder_landmark)

                #right wrist landmark
                rwrist_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                lwrist_landmark = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

                #left hand landmarks
                lindex_landmark = landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value]#left index finger landmark values
                lthumb_landmark = landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value]
                lpinky_landmark = landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value]

                #right hand landmarks
                rindex_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value]
                rthumb_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value]
                rpinky_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value]

                #knee landmark
                rknee_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                lknee_landmark = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]

                #ankle landmark
                rankle_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                lankle_landmark = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

                #foot index landmark
                lfoot_landmark = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
                rfoot_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]

                #foot heel landmark
                rheel_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value]
                lheel_landmark = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value]

                
                #elbow landmark
                relbow_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                lelbow_landmark = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]

                l_index_pos_np = np.array([lindex_landmark.x, lindex_landmark.y, lindex_landmark.z])#left hand
                l_elbow_pos_np = np.array([lelbow_landmark.x, lelbow_landmark.y, lelbow_landmark.z]) #left elbow
                l_shoulder_pos_np = np.array([lshoulder_landmark.x, lshoulder_landmark.y, lshoulder_landmark.z]) #left shoulder
                l_wrist_pos_np = np.array([lwrist_landmark.x, lwrist_landmark.y, lwrist_landmark.z]) #left wrist
                l_hip_pos_np = [lhip_landmark.x, lhip_landmark.y, lhip_landmark.z]
               
                
                #r_elbow_angle = calculate_angle_3d(np.array(rshoulder_landmark), np.array(relbow_landmark), np.array(rwrist_landmark))
                r_index_pos_np = np.array([rindex_landmark.x, rindex_landmark.y, rindex_landmark.z])
                r_elbow_pos_np = np.array([relbow_landmark.x, relbow_landmark.y, relbow_landmark.z])
                r_shoulder_pos_np = np.array([rshoulder_landmark.x, rshoulder_landmark.y, rshoulder_landmark.z])
                r_wrist_pos_np = np.array([rwrist_landmark.x, rwrist_landmark.y, rwrist_landmark.z])
                r_hip_pos_np = [rhip_landmark.x, rhip_landmark.y, rhip_landmark.z]
                
                #r_elbow_angle = calculate_angle_3d(np.array([rshoulder_landmark.x, rshoulder_landmark.y, rshoulder_landmark.z]),
                #np.array([relbow_landmark.x, relbow_landmark.y, relbow_landmark.z]), np.array([rwrist_landmark.x, rwrist_landmark.y, rwrist_landmark.z]))

                #print('right elbow angle : ', r_elbow_angle, '\t\tleft elbow angle : ', l_elbow_angle)
                l_elbow_angle = calculate_angle_3d(l_shoulder_pos_np, l_elbow_pos_np, l_wrist_pos_np) #LEFT elbow bending angle
                r_elbow_angle = calculate_angle_3d(r_shoulder_pos_np, r_elbow_pos_np, r_wrist_pos_np)

                #calculate left/right wrist angle
                l_wrist_angle = calculate_angle_3d(l_elbow_pos_np, l_wrist_pos_np, l_index_pos_np)#angle of left wrist
                r_wrist_angle = calculate_angle_3d(r_elbow_pos_np, r_wrist_pos_np, r_index_pos_np)

                

                #append wrist and elbow data
                l_wrist_angle_data.append(l_wrist_angle)
                r_wrist_angle_data.append(r_wrist_angle)

                r_elbow_angle_data.append(r_elbow_angle)
                l_elbow_angle_data.append(l_elbow_angle)

                #declare vectors for simpler calculations
                #foot_vec = np.array([global_x_right.x - global_x_left.x, global_x_right.z - global_x_left.z])
                foot_vec = np.array([rfoot_landmark.x - lfoot_landmark.x, rfoot_landmark.z - lfoot_landmark.z])
                #calculate hip turn
                hip_vec = np.array([rhip_landmark.x - lhip_landmark.x, rhip_landmark.z - lhip_landmark.z])
                shoulder_vec = np.array([rshoulder_landmark.x - lshoulder_landmark.x, rshoulder_landmark.z - lshoulder_landmark.z])
                #foot_vec = global_x_vec
                hip_turn_angle = calculate_body_turn(foot_vec, hip_vec)
                #left wrist thumb vector
                l_wrist_thumb_vec = [lthumb_landmark.x - lwrist_landmark.x, lthumb_landmark.y - lwrist_landmark.y, lthumb_landmark.z - lwrist_landmark.z]
                #right wrist thumb vector
                r_wrist_thumb_vec = [rthumb_landmark.x - rwrist_landmark.x, rthumb_landmark.y - rwrist_landmark.y, rthumb_landmark.z - rwrist_landmark.z]
                #right humerus vector
                r_hum_vec = np.array([relbow_landmark.x - rshoulder_landmark.x, relbow_landmark.y - rshoulder_landmark.y, relbow_landmark.z - rshoulder_landmark.z])
                #left humerus vector
                l_hum_vec = np.array([lelbow_landmark.x - lshoulder_landmark.x, lelbow_landmark.y - lshoulder_landmark.y, lelbow_landmark.z - lshoulder_landmark.z])
                #middle point of shoulders
                shoulder_middle = np.array([(rshoulder_landmark.x + lshoulder_landmark.x)/2, (rshoulder_landmark.y + lshoulder_landmark.y)/2, (rshoulder_landmark.z + lshoulder_landmark.z)/2])
                #middle point of hips
                hips_middle = np.array([(rhip_landmark.x + lhip_landmark.x)/2, (rhip_landmark.y + lhip_landmark.y)/2, (rhip_landmark.z + lhip_landmark.z)/2])
                #body vector
                body_vec = hips_middle - shoulder_middle
                #left body vector
                l_body_vec = [lhip_landmark.x - lshoulder_landmark.x, lhip_landmark.y - lshoulder_landmark.y, lhip_landmark.z - lshoulder_landmark.z]
                r_body_vec = [rhip_landmark.x - rshoulder_landmark.x, rhip_landmark.y - rshoulder_landmark.y, rhip_landmark.z - rshoulder_landmark.z]
                #upper leg vector
                r_femur_vec = np.array([rknee_landmark.x - rhip_landmark.x, rknee_landmark.y - rhip_landmark.y, rknee_landmark.z - rhip_landmark.z])
                l_femur_vec = np.array([lknee_landmark.x - lhip_landmark.x, lknee_landmark.y - lhip_landmark.y, lknee_landmark.z - lhip_landmark.z])
                #lower leg vector
                r_fibula_vec = np.array([rankle_landmark.x - rknee_landmark.x, rankle_landmark.y - rknee_landmark.y, rankle_landmark.z - rknee_landmark.z])
                l_fibula_vec = np.array([lankle_landmark.x - lknee_landmark.x, lankle_landmark.y - lknee_landmark.y, lankle_landmark.z - lknee_landmark.z])
                #foot direction vector
                r_footdir_vec = np.array([rfoot_landmark.x - rheel_landmark.x, rfoot_landmark.y - rheel_landmark.y, rfoot_landmark.z - rheel_landmark.z])
                l_footdir_vec = np.array([lfoot_landmark.x - lheel_landmark.x, lfoot_landmark.y - lheel_landmark.y, lfoot_landmark.z - lheel_landmark.z])
                #print('checkmark 1')
                
                ##calculating hip abduction/adduction
                #left hip abduction
                l_body_plane_a, l_body_plane_b, l_body_plane_c, l_body_plane_d = equation_plane(
                                                                    lhip_landmark.x, lhip_landmark.y, lhip_landmark.z,
                                                                    rhip_landmark.x, rhip_landmark.y, rhip_landmark.z,
                                                                    lshoulder_landmark.x, lshoulder_landmark.y, lshoulder_landmark.z)
                #project femur onto left body plane
                
                p1, p2, p3 = project_vector_onto_plane(l_femur_vec[0], l_femur_vec[1], l_femur_vec[2], 
                                                                    l_body_plane_a, l_body_plane_b, l_body_plane_c, l_body_plane_d)
                l_hip_frontal_angle = findangle([p1,p2,p3],l_body_vec)

                
                #print('checkmark 2')
                #right hip abduction
                r_body_plane_a, r_body_plane_b, r_body_plane_c, r_body_plane_d = equation_plane(
                                                                    rhip_landmark.x, rhip_landmark.y, rhip_landmark.z,
                                                                    lhip_landmark.x, lhip_landmark.y, lhip_landmark.z,
                                                                    rshoulder_landmark.x, rshoulder_landmark.y, rshoulder_landmark.z)
                p1, p2, p3 = project_vector_onto_plane(r_femur_vec[0], r_femur_vec[1], r_femur_vec[2],
                                                                    r_body_plane_a, r_body_plane_b, r_body_plane_c, r_body_plane_d)
                r_hip_frontal_angle = findangle([p1,p2,p3], r_body_vec)
                #print('checkmark 3')
                
                #calculate elbow horizontal 
                l_hand_plane_a, l_hand_plane_b, l_hand_plane_c, l_hand_plane_d = equation_plane(
                                                                    lindex_landmark.x, lindex_landmark.y, lindex_landmark.z,
                                                                    lthumb_landmark.x, lthumb_landmark.y, lthumb_landmark.z,
                                                                    lpinky_landmark.x, lpinky_landmark.y, lpinky_landmark.z)
                #print(l_hand_plane_a, l_hand_plane_b, l_hand_plane_c, l_hand_plane_d)
                l_elbow_horizontal_angle = angle_between_plane_and_vector(l_hand_plane_a, l_hand_plane_b, l_hand_plane_c, l_hand_plane_d,
                                                                    l_wrist_thumb_vec[0], l_wrist_thumb_vec[1], l_wrist_thumb_vec[2])
                #print('check')
                r_hand_plane_a, r_hand_plane_b, r_hand_plane_c, r_hand_plane_d = equation_plane(
                                                                    rindex_landmark.x, rindex_landmark.y, rindex_landmark.z,
                                                                    rthumb_landmark.x, rthumb_landmark.y, rthumb_landmark.z,
                                                                    rpinky_landmark.x, rpinky_landmark.y, rpinky_landmark.z)
                r_elbow_horizontal_angle = angle_between_plane_and_vector(r_hand_plane_a, r_hand_plane_b, r_hand_plane_c, r_hand_plane_d,
                                                                    r_wrist_thumb_vec[0], r_wrist_thumb_vec[1], r_wrist_thumb_vec[2])
                
                l_lowerbody_plane_a, l_lowerbody_plane_b, l_lowerbody_plane_c, l_lowerbody_plane_d = equation_plane(
                                                                    lhip_landmark.x, lhip_landmark.y, lhip_landmark.z,
                                                                    lknee_landmark.x, lknee_landmark.y, lknee_landmark.z,
                                                                    lankle_landmark.x, lankle_landmark.y, lankle_landmark.z
                )
                l_foot_plane_a, l_foot_plane_b, l_foot_plane_c, l_foot_plane_d = equation_plane(
                                                                    lankle_landmark.x, lankle_landmark.y, lankle_landmark.z,
                                                                    lheel_landmark.x, lheel_landmark.y, lheel_landmark.z,
                                                                    lfoot_landmark.x, lfoot_landmark.y, lfoot_landmark.z
                )
                l_ankle_frontal_angle = angle_between_two_planes(
                    l_lowerbody_plane_a, l_lowerbody_plane_b, l_lowerbody_plane_c, l_lowerbody_plane_d,
                    l_foot_plane_a, l_foot_plane_b, l_foot_plane_c, l_foot_plane_d
                )

                r_lowerbody_plane_a, r_lowerbody_plane_b, r_lowerbody_plane_c, r_lowerbody_plane_d = equation_plane(
                                                                    rhip_landmark.x, rhip_landmark.y, rhip_landmark.z,
                                                                    rknee_landmark.x, rknee_landmark.y, rknee_landmark.z,
                                                                    rankle_landmark.x, rankle_landmark.y, rankle_landmark.z
                )
                r_foot_plane_a, r_foot_plane_b, r_foot_plane_c, r_foot_plane_d = equation_plane(
                                                                    rankle_landmark.x, rankle_landmark.y, rankle_landmark.z,
                                                                    rheel_landmark.x, rheel_landmark.y, rheel_landmark.z,
                                                                    rfoot_landmark.x, rfoot_landmark.y, rfoot_landmark.z
                )
                r_ankle_frontal_angle = angle_between_two_planes(
                    r_lowerbody_plane_a, r_lowerbody_plane_b, r_lowerbody_plane_c, r_lowerbody_plane_d,
                    r_foot_plane_a, r_foot_plane_b, r_foot_plane_c, r_foot_plane_d
                )

                l_forearem_vec = [lwrist_landmark.x - lelbow_landmark.x, lwrist_landmark.y - lelbow_landmark.y, lwrist_landmark.z - lelbow_landmark.z]
                l_handspan_vec = [lindex_landmark.x - lthumb_landmark.x, lindex_landmark.y - lthumb_landmark.y, lindex_landmark.z - lthumb_landmark.z]

                l_wrist_frontal_angle = 90 - findangle(l_forearem_vec, l_handspan_vec)

                r_forearm_vec = [rwrist_landmark.x - relbow_landmark.x, rwrist_landmark.y - relbow_landmark.y, rwrist_landmark.z - relbow_landmark.z]
                r_handspan_vec = [rindex_landmark.x - rthumb_landmark.x, rindex_landmark.y - rthumb_landmark.y, rindex_landmark.z - rthumb_landmark.z]

                r_wrist_frontal_angle = 90 - findangle(r_forearm_vec, r_handspan_vec)

                l_wrist_frontal_angle_data.append(l_wrist_frontal_angle)
                r_wrist_frontal_angle_data.append(r_wrist_frontal_angle)

                l_hip_frontal_angle_data.append(l_hip_frontal_angle)
                r_hip_frontal_angle_data.append(r_hip_frontal_angle)


                l_elbow_horizontal_angle_data.append(l_elbow_horizontal_angle)
                r_elbow_horizontal_angle_data.append(r_elbow_horizontal_angle)
                #print('checkmark 4')

                l_ankle_frontal_angle_data.append(l_ankle_frontal_angle)
                r_ankle_frontal_angle_data.append(r_ankle_frontal_angle)


                #calculate ankle flexion extension
                r_ankle_sagittal = findangle(r_fibula_vec, r_footdir_vec)
                l_ankle_sagittal = findangle(l_fibula_vec, l_footdir_vec)
                #print('right ankle angle ', r_ankle_sagittal, '\t left ankle angle ', l_ankle_sagittal)
                r_ankle_angle_data.append(r_ankle_sagittal)
                l_ankle_angle_data.append(l_ankle_sagittal)
                

                #calculate knee flexion extension
                r_knee_sagittal = findangle(r_femur_vec, r_fibula_vec)
                l_knee_sagittal = findangle(l_femur_vec, l_fibula_vec)
                #print('checkmark 2')
                r_knee_angle_data.append(r_knee_sagittal)
                l_knee_angle_data.append(l_knee_sagittal)
                #print('right knee angle: ', r_knee_sagittal, '\tleft knee sagittal: ', l_knee_sagittal)
                
                #calculate abduction/adduction
                r_shoulder_frontal = findangle([r_hum_vec[0], r_hum_vec[1]], [body_vec[0], body_vec[1]]) #only use x,y
                l_shoulder_frontal = findangle([l_hum_vec[0], l_hum_vec[1]], [body_vec[0], body_vec[1]])
                #print('right frontal: ', r_shoulder_frontal, '\tl_shoulder_frontal', l_shoulder_frontal)
                r_should_front_data.append(r_shoulder_frontal)
                l_should_front_data.append(l_shoulder_frontal)

                #calculate flexion extenion
                r_shoulder_sagittal = findangle([r_hum_vec[1], r_hum_vec[2]], [body_vec[1], body_vec[2]]) #only use y,z
                l_shoulder_sagittal = findangle([l_hum_vec[1], l_hum_vec[2]], [body_vec[1], body_vec[2]])
                #print('right sagittal ', r_shoulder_sagittal)
                r_should_sag_data.append(r_shoulder_sagittal)
                l_should_sag_data.append(l_shoulder_sagittal)

                #calculate horizontal abduction/adduction
                r_shoulder_horizontal = findangle([r_hum_vec[0], r_hum_vec[2]], [body_vec[0], body_vec[2]]) #only use x,z
                l_shoulder_horizontal = findangle([l_hum_vec[0], l_hum_vec[2]], [body_vec[0], body_vec[2]])
                #print('right horizontal ', r_shoulder_horizontal, '\tleft horiziontal ',l_shoulder_horizontal)
                r_should_horiz_data.append(r_shoulder_horizontal)
                l_should_horiz_data.append(l_shoulder_horizontal)
                
                #calculate body turn
                #print('hip vec : ', hip_vec, '\t shoulder vec : ', shoulder_vec)
                body_turn_angle = calculate_body_turn(hip_vec, shoulder_vec)
                
                #print('body turn angle : ', body_turn_angle)
                
                #calculate hip flexion extension
                r_hip_sagittal = 180 - findangle([r_femur_vec[1], r_femur_vec[2]], [body_vec[1], body_vec[2]]) #use only y,z
                l_hip_sagittal = 180 - findangle([l_femur_vec[1], l_femur_vec[2]], [body_vec[1], body_vec[2]]) #use only y,z
                r_hip_sag_data.append(r_hip_sagittal)
                l_hip_sag_data.append(l_hip_sagittal)
                #print('rhip sagittal : ', r_hip_sagittal)
                #print('lhip sagittal : ', l_hip_sagittal)

                #calculate hip pronation supination
                r_hip_horizontal = 180 - findangle([r_femur_vec[0], r_femur_vec[1]], [body_vec[0], body_vec[1]]) # use only x,y
                l_hip_horizontal = 180 - findangle([l_femur_vec[0], l_femur_vec[1]], [body_vec[0], body_vec[1]]) # use only x,y
                r_hip_horiz_data.append(r_hip_horizontal)
                l_hip_horiz_data.append(l_hip_horizontal)
                #print('rhip horiz : ', r_hip_horizontal)
                #print('lhip horiz : ', l_hip_horizontal)

                #capture datas
                right_hip_position.append(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
                left_hip_position.append(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
                right_shoulder_position.append(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
                left_shoulder_position.append(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
                right_wrist_position.append(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
                left_wrist_position.append(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
                right_elbow_position.append(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
                left_elbow_position.append(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])
                
                #right/left knee
                right_knee_position.append(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
                left_knee_position.append(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])

                #right/left ankle
                right_ankle_position.append(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
                left_ankle_position.append(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

                #right/left foot
                right_foot_position.append(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value])
                left_foot_position.append(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value])

                
                hip_turn_data.append(hip_turn_angle)

                #print('body turn angle is : ', body_turn_angle)
                body_turn_data.append(body_turn_angle)
                elbow_angle_data.append(l_elbow_angle)

                
                #print("hip angle ", hip_turn_angle)
                #left_ankle_coord = (int(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x * width), int(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y * height))
                right_foot_coord = (int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * width), int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * height))
                #left_wrist_coord = (int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * width),  int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * height))
                #print("right foot y pos ", rfoot_landmark.y)
                
                #if((landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y < landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y) and swing_start_frame==0):
                #    swing_start_frame = currentframe

                if (body_turn_angle < max_clockwise_body_turn):
                    body_turn_angle = max_clockwise_body_turn
                    max_clockwise_body_turn_frame = currentframe
                    #max_clockwise_hip_vec = hip_vec
                    #max_clockwise_shoulder_vec = shoulder_vec
                    

                #print('ankle y pos : ', lankle_landmark.y)
                if(lankle_landmark.y < highest_foot_pos):
                    highest_foot_pos = lankle_landmark.y
                    swing_start_frame = currentframe
                    #print('swing start frame update')
                    
                else:
                    pass
                    #print('no update')
                #time.sleep(0.1)
                #print(lwrist_landmark.x)
                if(prev_wrist_max_pos < lwrist_landmark.x):
                    #print('end frame update')
                    prev_wrist_max_pos = lwrist_landmark.x
                    swing_end_frame = currentframe
                #image = cv2.circle(image, left_wrist_coord, 20, (255, 0, 0), 3)
                #image = cv2.circle(image, right_foot_coord, 20, (0, 255, 0), 3)
                #image = cv2.circle(image, (width*0.1, height*0.9), 20, (255,0,0), 3)

                if(lheel_landmark.x* width > config.box_start[0]):#if heel passes box start
                    form_condition_heel_passed_box_start = True
                    heel_passed_box_start_coordinate = lheel_landmark.x *  width #in pixels
                if(lheel_landmark.x * width > config.box_end[0]):#if heel passes box end
                    form_condition_heel_passed_box_end = True
                    toe_passed_box_end_coordinate = lheel_landmark.x * width #coordinate in pixels
                
                framenumber.append(currentframe)
                currentframe += 1
                #if(currentframe == 470):
                #    time.sleep(5)

        except:
            print('offsetframe increase')
            offset_frames+=1
            currentframe+=1
            pass
        
        #make detection
        results = pose.process(image)

        
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness = 2, circle_radius = 2),
                                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness = 2, circle_radius = 2) )

        
        cv2.rectangle(image, config.box_start, config.box_end, (0,255,0), 2)
        if(real_time):
            cv2.putText(image, "press q when finished", (10,450), font, 1, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow('Mediapipe feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
            break

        #if esc is pressed, start calculating 
        if cv2.waitKey(5) & 0xFF == 27:
            start = True
            print('starting recording')
            time.sleep(2)
    cap.release()
    cv2.destroyAllWindows()

    #ball_x_pos = np.array([data.x for data in ball_position])
    #ball_y_pos = np.array([-data.y for data in ball_position])
    #framenumber_ball = np.array([data.frame for data in ball_position])

    #fig, (axs1, axs2, ax3) = plt.subplots(3)
    #axs1.plot(framenumber_ball, ball_x_pos, '+')
    #axs1.set_title("x pos")

    #axs2.plot(framenumber_ball, ball_y_pos, 'o')
    #axs2.set_title("y pos")

    #ax3.plot(ball_x_pos, ball_y_pos, 'o')
    #ax3.set_title("xypos")

    #Highlighter = Highlighter(ax3, ball_x_pos, ball_y_pos)
    #plt.show()
    #selected_region = Highlighter.mask
    #print(ball_x_pos[selected_region], ball_y_pos[selected_region])
    
    #selected_x = ball_x_pos[selected_region]
    #selected_y = ball_y_pos[selected_region]

    #numpoints = len(selected_x)
    #avg_delta_x = np.average(selected_x)
    #avg_delta_y = np.average(selected_y)
    #avg_angle = np.arctan(avg_delta_y/avg_delta_x)*(180/math.pi)
    print('==============================')
    #print('avg ball angle is : ', avg_angle)
    #selected_frames = framenumber_ball[selected_region]
    """
    print('selected frames: ', selected_frames)
    print(selected_x)
    dx = np.diff(selected_x)
    dx = savgol_filter(dx, 5,3)
    dy = np.diff(selected_y)
    dy = savgol_filter(dy, 5,3)
    plt.plot(np.arange(0,dx.size,1), dx)
    plt.plot(np.arange(0,dy.size,1), dy)
    plt.xlabel('frames')
    plt.ylabel('dx')
    
    print (dy)
    print(dx, dy)
    print(np.shape(dx))
    print(np.shape(np.stack((dx, dy), axis=0)))
    dframe = np.diff(selected_frames)
    dists = LA.norm(np.stack((dx, dy)), axis = 0)
    plt.plot(np.arange(0, dists.size,1), dists)
    plt.show()
    dframe[dframe==0] = 1
    #print(np.shape(dists))
    print('*********dframe*********', dframe)
    speeds = dists/dframe
    avg_speed = np.average(speeds)
    print('avg speed in pixels per frame is :', avg_speed)
    avg_speed_meter_per_second = avg_speed * meter_per_pixel * config.frame_per_second
    print('avg speed in meter per second is : ', avg_speed_meter_per_second)


    angle = []
    speed = []
    """
    
    
    #find backswing start frame
    ankle_delta = left_ankle_position[0].y - left_ankle_position[swing_start_frame].y
    backswing_start_y = left_ankle_position[0].y - ankle_delta*0.2
    
    #print('ankle delta : ', ankle_delta)
    #print('backswing start y pos : ', backswing_start_y)
    for index in range(swing_start_frame, -1, -1):
        if(left_ankle_position[index].y < backswing_start_y):
            continue #ignore number, need to find number that is greater than our target ankle pos y
        else:
            backswing_start_frame = index
            break

    #print('backswing start frame : ', backswing_start_frame)
    #print('swing start frame : ', swing_start_frame)
    #print('swing end frame : ', swing_end_frame)
    #print('ball pos start : ', ball_position[0].frame)

    """
    trimmed_xy = ball_position[:swing_end_frame-ball_position[0].frame]
    print('trimmed xy : ', len(ball_position))
    #trimmed_y = ball_position[:swing_end_frame-ball_position[0].frame]
    for index in range(len(trimmed_xy)-2):
        delta_y = -(trimmed_xy[index+1].y - trimmed_xy[index].y)
        delta_x = trimmed_xy[index+1].x - trimmed_xy[index].x
        if delta_x == 0:
            continue
        angle.append(np.arctan(delta_y/delta_x)*(180/math.pi))
        #print("index+1 frame", trimmed_xy[index+1].frame)
        #print("index frame", trimmed_xy[index].frame)
        #print("")
        if (trimmed_xy[index+1].frame != trimmed_xy[index].frame):
            speed.append(np.sqrt(delta_x**2 + delta_y**2)/((trimmed_xy[index+1].frame - trimmed_xy[index].frame)))
    
        

    #angle = [np.arctan(grad_x/grad_y)]
    print("==================")
    print("공 각도 (degrees) ", np.average(angle))
    print("공 속도 (pixels/frame) ", np.average(speed))
    print("")
    """

    """
    hip_pos = np.zeros((3, len(hip_position)))
    shoulder_pos = np.zeros((3, len(shoulder_position)))
    wrist_pos = np.zeros((3, len(wrist_position)))
    index=0
    print(np.shape(hip_turn_data))
    for hip_data in hip_position:
        new_col = [hip_data.x, hip_data.y, hip_data.z]
        hip_pos[:, index] = new_col
        index+=1
    index = 0
    for shoulder_data in shoulder_position:
        new_col = [shoulder_data.x, shoulder_data.y, shoulder_data.z]
        shoulder_pos[:, index] = new_col
        index+=1
    index=0
    for wrist_data in wrist_position:
        new_col = [wrist_data.x, wrist_data.y, wrist_data.z]
        wrist_pos[:, index] = new_col
        index+=1
    """

    #if smooth, aka framerate always increments by 1, then ok to do gradient
    smooth = True
    for x in range(len(framenumber)-1):
        if framenumber[x+1]-framenumber[x]==1:
            continue
        else:
            #smooth = False
            break
    
    #if data is smooth, calculate acceleration
    if True:
        """
        hip_accel = np.gradient(np.gradient(hip_pos, axis=1), axis=1)
        shoulder_accel = np.gradient(np.grfdient(shoulder_pos, axis=1), axis=1)
        wrist_accel = np.gradient(np.gradient(wrist_pos, axis=1), axis=1)
        
        hip_accel_norm = LA.norm(hip_accel, axis=0)
        shoulder_accel_norm = LA.norm(shoulder_accel, axis=0)
        wrist_accel_norm = LA.norm(wrist_accel, axis=0)
        """
        #print('body turn data length : ', len(body_turn_data))
        torso_angular_vel = np.gradient(body_turn_data)
        hip_angular_vel = np.gradient(hip_turn_data)
        wrist_angular_vel = np.gradient(l_wrist_angle_data)
        elbow_angular_vel = np.gradient(elbow_angle_data)

        
        torso_angular_vel_max = np.argmax(torso_angular_vel)
        hip_angular_vel_max = np.argmax(hip_angular_vel)
        wrist_angular_vel_max = np.argmax(wrist_angular_vel)

        hip_turn_data_smooth = savgol_filter(hip_turn_data, 11, 3)
        hip_angular_vel_smooth = np.gradient(hip_turn_data_smooth, 3)

        torso_turn_data_smooth = savgol_filter(body_turn_data, 11, 3)
        torso_angular_vel_smooth = np.gradient(torso_turn_data_smooth, 3)

        wrist_angle_data_smooth = savgol_filter(l_wrist_angle_data, 11, 3)
        wrist_angular_vel_smooth = np.gradient(wrist_angle_data_smooth, 10)

        elbow_angle_data_smooth = savgol_filter(elbow_angle_data, 11, 3)
        elbow_angular_vel_smooth = np.gradient(elbow_angle_data_smooth, 3)
        

        swing_length = swing_end_frame - swing_start_frame

        #운동수행 폼 #1 condition
        impactframe = np.argmax(elbow_angular_vel_smooth[swing_start_frame : swing_end_frame]) + offset_frames + swing_start_frame
        maximum_backswing_frames = 0.455 * config.frame_per_second
        #print('impact frame: ', impactframe)
        #print('max clockwise body frame: ', max_clockwise_body_turn_frame)
        
        if(impactframe - max_clockwise_body_turn_frame < maximum_backswing_frames):
            form_condition_backswing=True
        #print(np.argmax(hip_angular_vel_smooth[swing_start_frame : swing_end_frame]))

        #운동수행 폼 #2 condition
        if(form_condition_heel_passed_box_start and not form_condition_heel_passed_box_end):
            form_condition_foot = True

        #운동수행 폼 #3 condition
        if(hip_turn_data_smooth[impactframe] - hip_turn_data_smooth[max_clockwise_body_turn_frame] > 80.3 and 
        torso_turn_data_smooth[impactframe] - torso_turn_data_smooth[max_clockwise_body_turn_frame] > 102.8):
            form_condition_forwardswing = True

        max_torso_speed_frame = np.argmax(torso_angular_vel_smooth[swing_start_frame : swing_end_frame]) + offset_frames + swing_start_frame
        max_hip_speed_frame = np.argmax(hip_angular_vel_smooth[swing_start_frame : swing_end_frame]) + offset_frames + swing_start_frame

        #운동수행 폼 #4 condition
        if(impactframe > max_torso_speed_frame and max_torso_speed_frame > max_hip_speed_frame):
            form_condition_sequence=True

        print('form conditions')
        print(form_condition_backswing, form_condition_foot, form_condition_forwardswing, form_condition_sequence)

        if(form_condition_backswing == False):
            print(backswing_fail_string)
        elif(form_condition_foot == False):
            print(foot_fail_string)
        elif(form_condition_forwardswing == False):
            print(forwardswing_fail_string)
        elif(form_condition_sequence == False):
            print(sequence_fail_string)
        else:
            print(success_string)
        #print('avg ball angle is : ', avg_angle)

        if(config.verbose):
            print("==========1번=============")
            print("1st condition is ", form_condition_backswing)
            print("maximum allows backswing frames : ", maximum_backswing_frames)
            print("impactframe : ", impactframe)
            print("maximum clockwise body turn frame : ", max_clockwise_body_turn_frame)
            print("frames taken for backswing : ", impactframe - max_clockwise_body_turn_frame)
            print("==========2번=============")
            print("2nd condition is ", form_condition_foot)
            if(form_condition_heel_passed_box_start):
                print("heel passed box by : ", (heel_passed_box_start_coordinate - config.box_start[0]) *  meter_per_pixel)
            if(not form_condition_heel_passed_box_start):
                print("heel did not pass box by : ", (config.box_start[0] - heel_passed_box_start_coordinate) * meter_per_pixel)
            if(form_condition_heel_passed_box_end):
                print("heel passed box end by : ", (toe_passed_box_end_coordinate - config.box_end[0]) * meter_per_pixel)
            if(not form_condition_heel_passed_box_end):
                print("heel did not pass box end by : ", -(toe_passed_box_end_coordinate - config.box_end[0]) * meter_per_pixel)
            print("=========3번============")
            print("3rd condition is ", form_condition_forwardswing)
            print("hip turn angle at maximum backswing : ", hip_turn_data_smooth[max_clockwise_body_turn_frame])
            print("hip turn angle at impact : ", hip_turn_data_smooth[impactframe])
            print("torso turn angle at maximum backswing : ", torso_turn_data_smooth[max_clockwise_body_turn])
            print("torso turn agnle at impact : ", torso_turn_data_smooth[impactframe])
            print("===========4번==========")
            print("maximum hip turn speed frame : ", max_hip_speed_frame )
            print("maximum torso turn speed frame : ", max_torso_speed_frame)
            print("maximum elbow turn speed frame : ", impactframe)

        

        """
        print("엉덩관절 최대 회전 각속도 @ frame number ", max_hip_speed_frame)
        print((np.argmax(hip_angular_vel_smooth[swing_start_frame : swing_end_frame]))/swing_length * 100.0, "%")
        print("")

        print("몸통 최대 각속도 @ frame number ", max_torso_speed_frame)
        print((np.argmax(torso_angular_vel_smooth[swing_start_frame : swing_end_frame]))/swing_length * 100.0, "%")
        print("")

        print("팔꿈치 최대 각속도 @ frame number", impactframe)
        print((np.argmax(elbow_angular_vel_smooth[swing_start_frame : swing_end_frame]))/swing_length * 100.0, "%")
        print("")

        
        print("손목 최대 각속도 @ frame number ", np.argmax(wrist_angular_vel_smooth[swing_start_frame : swing_end_frame]) + offset_frames + swing_start_frame)
        print((np.argmax(wrist_angular_vel_smooth[swing_start_frame : swing_end_frame]))/swing_length * 100.0, "%")
        print("")
        """


        #print("swing start at frame ", swing_start_frame + offset_frames)
        #print('swing end at frame ', swing_end_frame + offset_frames)
        #print('offset frames', offset_frames)
        framenumber = [a + offset_frames for a in framenumber]
        if(config.make_excel):
            result_name = 'poseall_result_{}.xlsx'.format(video_name)
            coordinate_name = 'poseall_coordinate_{}.xlsx'.format(video_name)
            angle_name = 'poseall_body_angle_{}.xlsx'.format(video_name)
            #ball_name = 'ball_angle_{}.xlsx'.format(video_name)

            #df = DataFrame({'ball angle' : [avg_angle], 'ball speed (pixels/frame)' : [avg_speed], 'ball speed (meters/second)' : [avg_speed_meter_per_second]})
            #df.to_excel(ball_name, sheet_name='sheet1', index=False)

            df = DataFrame({'Frame': framenumber, 
                            'hip angle no smoothing': hip_turn_data, 'hip angle with smoothing': hip_turn_data_smooth, 'hip angular velocity (from smooth)': hip_angular_vel_smooth,
                            'torso angle no smoothing': body_turn_data, 'torso angle with smoothing': torso_turn_data_smooth, 'torso angular velocity (from smooth)': torso_angular_vel_smooth,
                            'wrist angle no smoothing': l_wrist_angle_data, 'wrist angle with smoothing': wrist_angle_data_smooth, 'wrist angular velocity (from smooth)' : wrist_angular_vel_smooth,
                            'elbow angle no smoothing': elbow_angle_data, 'elbow angle with smoothing': elbow_angle_data_smooth, 'elbow angular velocity(from smooth)': elbow_angular_vel_smooth
                            })
            df.to_excel(result_name, sheet_name='sheet1', index=False)
            
            df = DataFrame({'Frame' : framenumber,
                            'right hip x' : [data.x for data in right_hip_position], 'right hip y' : [data.y for data in right_hip_position], 'right hip z' : [data.z for data in right_hip_position],
                            'left hip x' : [data.x for data in left_hip_position], 'left hip y' : [data.y for data in left_hip_position], 'left hip z' : [data.z for data in left_hip_position],

                            'right shoulder x' : [data.x for data in right_shoulder_position], 'right shoulder y' : [data.y for data in right_shoulder_position], 'right shoulder z' : [data.z for data in right_shoulder_position], 
                            'left shoulder x' : [data.x for data in left_shoulder_position], 'left shoulder y' : [data.y for data in left_shoulder_position], 'left shoulder z' : [data.z for data in left_shoulder_position],

                            'right wrist x' : [data.x for data in right_wrist_position], 'right wrist y' : [data.y for data in right_wrist_position], 'right wrist z' : [data.z for data in right_wrist_position],
                            'left wrist x' : [data.x for data in left_wrist_position], 'left wrist y' : [data.y for data in left_wrist_position], 'left wrist z' : [data.z for data in left_wrist_position],

                            'right elbow x' : [data.x for data in right_elbow_position], 'right elbow y' : [data.y for data in right_elbow_position], 'right elbow z' : [data.z for data in right_elbow_position],
                            'left elbow x' : [data.x for data in left_elbow_position], 'left elbow y' : [data.y for data in left_elbow_position], 'left elbow z' : [data.z for data in left_elbow_position],

                            'right knee x' : [data.x for data in right_knee_position], 'right knee y' : [data.y for data in right_knee_position], 'right knee z' : [data.z for data in right_knee_position],
                            'left knee x' : [data.x for data in left_knee_position], 'left knee y' : [data.y for data in left_knee_position], 'left knee z' : [data.z for data in left_knee_position],

                            'right ankle x' : [data.x for data in right_ankle_position], 'right ankle y' : [data.y for data in right_ankle_position], 'right ankle z' : [data.z for data in right_ankle_position],
                            'left ankle x' : [data.x for data in left_ankle_position], 'left ankle y' : [data.y for data in left_ankle_position], 'left ankle z' : [data.z for data in left_ankle_position],

                            'right foot x' : [data.x for data in right_foot_position], 'right foot y' : [data.y for data in right_foot_position], 'right foot z': [data.z for data in right_foot_position],
                            'left foot x' : [data.x for data in left_foot_position], 'left foot y' : [data.y for data in left_foot_position], 'left foot z' : [data.z for data in left_foot_position]
                            })
            df.to_excel(coordinate_name, sheet_name= 'sheet1', index=False)

            df = DataFrame({'right ankle sagittal' : r_ankle_angle_data, 'left ankle sagittal' : l_ankle_angle_data,
                            'right elbow horizontal' : r_elbow_horizontal_angle_data, 'left elbow horizontal' : l_elbow_horizontal_angle_data,
                            'right elbow sagittal' : r_elbow_angle_data, 'left elbow sagittal' : l_elbow_angle_data,
                            'right wrist sagittal' : r_wrist_angle_data, 'left wrist sagittal' : l_wrist_angle_data,
                            'right wrist frontal' : r_wrist_frontal_angle_data, 'left wrist frontal' : l_wrist_frontal_angle_data,
                            'right ankle frontal' : r_ankle_frontal_angle_data, 'left ankle frontal' : l_ankle_frontal_angle_data,
                            'right hip frontal' : r_hip_frontal_angle_data, 'left hip frontal' : l_hip_frontal_angle_data,
                            'right knee sagittal' : r_knee_angle_data, 'left knee sagittal' : l_knee_angle_data,
                            'right shoulder sagittal' : r_should_sag_data, 'left shoulder sagittal' : l_should_sag_data,
                            'right shoulder horiz' : r_should_horiz_data, 'left shoulder horiz' : l_should_horiz_data,
                            'right shoulder frontal' : r_should_front_data, 'left shoulder frontal' : l_should_front_data,
                            'body turn' : body_turn_data, 'hip turn' : hip_turn_data, 'wrist angle' : l_wrist_angle_data,
                            'right hip sagittal' : r_hip_sag_data, 'left hip sagittal' : l_hip_sag_data,
                            'right hip horizontal' : r_hip_horiz_data, 'left hip horizontal': l_hip_horiz_data
                            })
            df.to_excel(angle_name, sheet_name = 'sheet1', index = False)
        """
        fig, axs = plt.subplots(3 ,4)
        axs[0,0].plot(framenumber, hip_turn_data_smooth)
        axs[0,0].set_title('hip turn smooth')

        axs[1,0].plot(framenumber, hip_turn_data)
        axs[1,0].set_title('hip turn')

        axs[2,0].plot(framenumber, hip_angular_vel_smooth)
        axs[2,0].set_title('hip angular velocity')


        axs[0,1].plot(framenumber, torso_turn_data_smooth)
        axs[0,1].set_title('torso turn smooth')

        axs[1,1].plot(framenumber, body_turn_data)
        axs[1,1].set_title('torso turn')

        axs[2,1].plot(framenumber, torso_angular_vel_smooth)
        axs[2,1].set_title('torso angular velocity')


        axs[0,2].plot(framenumber, elbow_angle_data_smooth)
        axs[0,2].set_title('elbow angle smooth')

        axs[1,2].plot(framenumber, elbow_angle_data)
        axs[1,2].set_title('elbow angle')

        axs[2,2].plot(framenumber, elbow_angular_vel_smooth)
        axs[2,2].set_title('elbow angular velocity')


        axs[0,3].plot(framenumber, wrist_angle_data_smooth)
        axs[0,3].set_title('wrist angle smooth')

        axs[1,3].plot(framenumber, l_wrist_angle_data)
        axs[1,3].set_title('wrist angle')

        axs[2,3].plot(framenumber, wrist_angular_vel_smooth)
        axs[2,3].set_title('wrist angular velocity')

        
        hip_angvel_graph = plt.figure(0)
        plt.scatter(framenumber, hip_angular_vel_smooth)
        plt.title("hip angular velocity with smoothing")
        
        torso_angvel_graph = plt.figure(1)
        plt.scatter(framenumber, torso_angular_vel_smooth)
        plt.title("torso angular velocity with smoothing")
        wrist_angvel_graph = plt.figure(2)
        plt.scatter(framenumber, wrist_angular_vel_smooth)
        plt.title("wrist angular velocity with smoothing")
        elbow_angvel_graph = plt.figure(3)
        plt.scatter(framenumber, elbow_angular_vel_smooth)
        plt.title("elbow angular velocity with smoothing")
        hip_turn_graph = plt.figure(4)
        plt.scatter(framenumber, hip_turn_data_smooth)
        plt.scatter(framenumber, hip_turn_data)
        plt.title("hip turn blue = smoothed")
        torso_turn_graph = plt.figure(6)
        plt.scatter(framenumber, torso_turn_data_smooth)
        plt.scatter(framenumber, body_turn_data)
        plt.title("torso turn blue = smoothed")
        elbow_angle_graph = plt.figure(7)
        plt.scatter(framenumber, elbow_angle_data_smooth)
        plt.scatter(framenumber, elbow_angle_data)
        plt.title("elbow angle data blue = smoothed")
        wrist_angle_graph = plt.figure(8)
        plt.scatter(framenumber, wrist_angle_data_smooth)
        plt.scatter(framenumber, wrist_angle_data)
        plt.title("wrist angle data blue = smoothed")
        """
        plt.show()
    else:
        print('camera missed some frames and was not able to do analysis')
