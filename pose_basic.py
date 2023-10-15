import cv2
#from matplotlib import offsetbox
#from matplotlib.text import OffsetFrom
import mediapipe as mp
import numpy as np
from numpy import linalg as LA, r_
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import time
import math
from pandas import DataFrame

#CHANGE VIDEO NAME 
#프로그렘 돌리기전에 파일 이름 바꿔야됨
#파일 타입 무조건 포함해야됨 예) ".mp4"
video_name = "squat.mp4."
CAMERA_FRAMREATE = 60
#시뇌값
pose_confidence_value = 0.9

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

offset_frames = 0

prev_elbow_angle = 0
prev_wrist_angle = 0

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

wrist_angle_data = []
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

ball_position = []


#start = False
start = True
do_once = True

class detection:
    def __init__(self, frame, x, y):
        self.frame = frame
        self.x = x
        self.y = y

#def ball_tracking(ret, frame, frame_next, position, currentframe):
#    if not ret:
#        print("ball cannot be tracked")
#        return
#    diff = cv2.absdiff(frame, frame_next)
#    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
#    blur = cv2.GaussianBlur(gray, (5,5), 0)
#    _, movement_mask = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY) 
#
#    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#    lowerhsv = np.array([0, 177, 87])
#    upperhsv = np.array([40, 255, 255])
#
#    color_mask = cv2.inRange(hsv, lowerhsv, upperhsv)\
#    
#    combined_mask = cv2.bitwise_and(movement_mask, color_mask)
#    result = cv2.bitwise_and(frame, frame, mask = combined_mask)
#    dilated = cv2.dilate(combined_mask, None, iterations = 3)
#    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#    for contour in contours:
#        if cv2.contourArea(contour) < 300:
#             continue
#        (x,y), radius=cv2.minEnclosingCircle(contour)
#        center=(int(x), int(y))
#        radius=int(radius)
#        cv2.circle(result, center, radius, (0, 255, 0), 2)
#        position.append(detection(currentframe, x, y))
#
#    cv2.imshow("frame", frame)
#    cv2.imshow("color mask", color_mask)
#    cv2.imshow("movement mask", movement_mask)
#    cv2.imshow("result", result)
    
    #currentframe += 1

def calculate_body_turn(shoulder_vec, hip_vec):
    v1=shoulder_vec/LA.norm(shoulder_vec)
    v2=hip_vec/LA.norm(hip_vec)
    res = np.dot(v1, v2)
    #print('dot prod for body is : ', res)
    angle_rad = np.arccos(res)
    #print('body angle : ', math.degrees(angle_rad))
    #if(np.cross(v1, v2)<0):
    #    return -1*math.degrees(angle_rad)
    return math.degrees(angle_rad)

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
    cap = cv2.VideoCapture(video_name)
    
    while cap.isOpened():
        width  = cap.get(3)
        height = cap.get(4)
        ret, frame = cap.read()
        ret, frame_next = cap.read()
        
        

        if not ret:
            print("can't receive frame. exiting")
            break

        #ball_tracking(ret, frame, frame_next, ball_position, currentframe)

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

                
                #calculate elbow angle
                relbow_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                lelbow_landmark = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                elbow_pos_np = np.array([lelbow_landmark.x, lelbow_landmark.y, lelbow_landmark.z]) #left elbow
                shoulder_pos_np = np.array([lshoulder_landmark.x, lshoulder_landmark.y, lshoulder_landmark.z]) #left shoulder
                wrist_pos_np = np.array([lwrist_landmark.x, lwrist_landmark.y, lwrist_landmark.z]) #left wrist
                l_elbow_angle = calculate_angle_3d(shoulder_pos_np, elbow_pos_np, wrist_pos_np) #LEFT elbow angle
                #r_elbow_angle = calculate_angle_3d(np.array(rshoulder_landmark), np.array(relbow_landmark), np.array(rwrist_landmark))

                r_elbow_angle = calculate_angle_3d(np.array([rshoulder_landmark.x, rshoulder_landmark.y, rshoulder_landmark.z]),
                np.array([relbow_landmark.x, relbow_landmark.y, relbow_landmark.z]), np.array([rwrist_landmark.x, rwrist_landmark.y, rwrist_landmark.z]))

                #print('right elbow angle : ', r_elbow_angle, '\t\tleft elbow angle : ', l_elbow_angle)
                r_elbow_angle_data.append(r_elbow_angle)
                l_elbow_angle_data.append(l_elbow_angle)

                
                #calculate wrist angle
                hand_landmark = landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value]#left index finger landmark values
                hand_pos_np = np.array([hand_landmark.x, hand_landmark.y, hand_landmark.z])#left hand
                wrist_angle = calculate_angle_3d(elbow_pos_np, wrist_pos_np, hand_pos_np)#angle of left wrist
                
                #foot_vec = np.array([global_x_right.x - global_x_left.x, global_x_right.z - global_x_left.z])
                foot_vec = np.array([rfoot_landmark.x - lfoot_landmark.x, rfoot_landmark.z - lfoot_landmark.z])
                #calculate hip turn
                hip_vec = np.array([rhip_landmark.x - lhip_landmark.x, rhip_landmark.z - lhip_landmark.z])
                shoulder_vec = np.array([rshoulder_landmark.x - lshoulder_landmark.x, rshoulder_landmark.z - lshoulder_landmark.z])
                #foot_vec = global_x_vec
                hip_turn_angle = calculate_body_turn(foot_vec, hip_vec)

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

                wrist_angle_data.append(wrist_angle)
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

                #print('ankle y pos : ', lankle_landmark.y)
                #if(lankle_landmark.y < highest_foot_pos):
                #    highest_foot_pos = lankle_landmark.y
                #    swing_start_frame = currentframe
                #    print('swing start frame update')
                #    
                #else:
                #    print('no update')
                #time.sleep(0.1)
                #print(lwrist_landmark.x)
                #if(prev_wrist_max_pos < lwrist_landmark.x):
                #    #print('end frame update')
                #    prev_wrist_max_pos = lwrist_landmark.x
                #    swing_end_frame = currentframe
                #image = cv2.circle(image, left_wrist_coord, 20, (255, 0, 0), 3)
                #image = cv2.circle(image, right_foot_coord, 20, (0, 255, 0), 3)
                #image = cv2.circle(image, (width*0.1, height*0.9), 20, (255,0,0), 3)
                
                framenumber.append(currentframe)
                currentframe += 1
                #if(currentframe == 470):
                #    time.sleep(5)

        except:
            #print('offsetframe increase')
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

    #ball_x_pos = [data.x for data in ball_position]
    #ball_y_pos = [-data.y for data in ball_position]
    #framenumber_ball = [data.frame for data in ball_position]

    #fig, (axs1, axs2, ax3) = plt.subplots(3)
    #axs1.plot(framenumber_ball, ball_x_pos, '+')
    #axs1.set_title("x pos")

    #axs2.plot(framenumber_ball, ball_y_pos, 'o')
    #axs2.set_title("y pos")

    #ax3.plot(ball_x_pos, ball_y_pos, 'o')
    #ax3.set_title("xypos")

    #angle = []
    #speed = []
    
    #find backswing start frame
    #ankle_delta = left_ankle_position[0].y - left_ankle_position[swing_start_frame].y
    #backswing_start_y = left_ankle_position[0].y - ankle_delta*0.2

    #print('ankle delta : ', ankle_delta)
    #print('backswing start y pos : ', backswing_start_y)
    #for index in range(swing_start_frame, -1, -1):
    #    if(left_ankle_position[index].y < backswing_start_y):
    #        continue #ignore number, need to find number that is greater than our target ankle pos y
    #    else:
    #        backswing_start_frame = index
    #        break

    #print('backswing start frame : ', backswing_start_frame)
    #print('swing start frame : ', swing_start_frame)
    #print('swing end frame : ', swing_end_frame)
    #print('ball pos start : ', ball_position[0].frame)
    #trimmed_xy = ball_position[:swing_end_frame-ball_position[0].frame]
    #print('trimmed xy : ', len(ball_position))
    ##trimmed_y = ball_position[:swing_end_frame-ball_position[0].frame]
    #for index in range(len(trimmed_xy)-2):
    #    delta_y = -(trimmed_xy[index+1].y - trimmed_xy[index].y)
    #    delta_x = trimmed_xy[index+1].x - trimmed_xy[index].x
    #    if delta_x == 0:
    #        continue
    #    angle.append(np.arctan(delta_y/delta_x)*(180/math.pi))
    #    print("index+1 frame", trimmed_xy[index+1].frame)
    #    print("index frame", trimmed_xy[index].frame)
    #    print("")
    #    if (trimmed_xy[index+1].frame != trimmed_xy[index].frame):
    #        speed.append(np.sqrt(delta_x**2 + delta_y**2)/((trimmed_xy[index+1].frame - trimmed_xy[index].frame)))
       
        

    #angle = [np.arctan(grad_x/grad_y)]
    #print("==================")
    #print("공 각도 (degrees) ", np.average(angle))
    #print("공 속도 (pixels/frame) ", np.average(speed))
    #print("")

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
        wrist_angular_vel = np.gradient(wrist_angle_data)
        elbow_angular_vel = np.gradient(elbow_angle_data)

        
        torso_angular_vel_max = np.argmax(torso_angular_vel)
        hip_angular_vel_max = np.argmax(hip_angular_vel)
        wrist_angular_vel_max = np.argmax(wrist_angular_vel)

        hip_turn_data_smooth = savgol_filter(hip_turn_data,19, 3)
        hip_angular_vel_smooth = np.gradient(hip_turn_data_smooth, 3)

        torso_turn_data_smooth = savgol_filter(body_turn_data, 19, 3)
        torso_angular_vel_smooth = np.gradient(torso_turn_data_smooth, 3)

        wrist_angle_data_smooth = savgol_filter(wrist_angle_data, 19, 3)
        wrist_angular_vel_smooth = np.gradient(wrist_angle_data_smooth, 10)

        elbow_angle_data_smooth = savgol_filter(elbow_angle_data, 19, 3)
        elbow_angular_vel_smooth = np.gradient(elbow_angle_data_smooth, 3)
        

        #swing_length = swing_end_frame - swing_start_frame
        
        #print(np.argmax(hip_angular_vel_smooth[swing_start_frame : swing_end_frame]))

        #print("엉덩관절 최대 회전 각속도 @ frame number ", np.argmax(hip_angular_vel_smooth[swing_start_frame : swing_end_frame]) + offset_frames + swing_start_frame)
        #print((np.argmax(hip_angular_vel_smooth[swing_start_frame : swing_end_frame]))/swing_length * 100.0, "%")
        #print("")

        #print("몸통 최대 각속도 @ frame number ", np.argmax(torso_angular_vel_smooth[swing_start_frame : swing_end_frame]) + offset_frames + swing_start_frame)
        #print((np.argmax(torso_angular_vel_smooth[swing_start_frame : swing_end_frame]))/swing_length * 100.0, "%")
        #print("")

        #print("팔꿈치 최대 각속도 @ frame number", np.argmax(elbow_angular_vel_smooth[swing_start_frame : swing_end_frame]) + offset_frames + swing_start_frame)
        #print((np.argmax(elbow_angular_vel_smooth[swing_start_frame : swing_end_frame]))/swing_length * 100.0, "%")
        #print("")

        #print("손목 최대 각속도 @ frame number ", np.argmax(wrist_angular_vel_smooth[swing_start_frame : swing_end_frame]) + offset_frames + swing_start_frame)
        #print((np.argmax(wrist_angular_vel_smooth[swing_start_frame : swing_end_frame]))/swing_length * 100.0, "%")
        #print("")


        #print("swing start at frame ", swing_start_frame + offset_frames)
        #print('swing end at frame ', swing_end_frame + offset_frames)
        #print('offset frames', offset_frames)
        framenumber = [a + offset_frames for a in framenumber]

        result_name = 'result_{}.xlsx'.format(video_name)
        coordinate_name = 'coordinate_{}.xlsx'.format(video_name)
        angle_name = 'angle_{}.xlsx'.format(video_name)

        df = DataFrame({'Frame': framenumber, 
                        'hip angle no smoothing': hip_turn_data, 'hip angle with smoothing': hip_turn_data_smooth, 'hip angular velocity (from smooth)': hip_angular_vel_smooth,
                        'torso angle no smoothing': body_turn_data, 'torso angle with smoothing': torso_turn_data_smooth, 'torso angular velocity (from smooth)': torso_angular_vel_smooth,
                        'wrist angle no smoothing': wrist_angle_data, 'wrist angle with smoothing': wrist_angle_data_smooth, 'wrist angular velocity (from smooth)' : wrist_angular_vel_smooth,
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
                        'right elbow sagittal' : r_elbow_angle_data, 'left elbow sagittal' : l_elbow_angle_data,
                        'right knee sagittal' : r_knee_angle_data, 'left knee sagittal' : l_knee_angle_data,
                        'right shoulder sagittal' : r_should_sag_data, 'left shoulder sagittal' : l_should_sag_data,
                        'right shoulder horiz' : r_should_horiz_data, 'left shoulder horiz' : l_should_horiz_data,
                        'right shoulder frontal' : r_should_front_data, 'left shoulder frontal' : l_should_front_data,
                        'body turn' : body_turn_data, 'hip turn' : hip_turn_data, 'wrist angle' : wrist_angle_data,
                        'right hip sagittal' : r_hip_sag_data, 'left hip sagittal' : l_hip_sag_data,
                        'right hip horizontal' : r_hip_horiz_data, 'left hip horizontal': l_hip_horiz_data
                        })
        df.to_excel(angle_name, sheet_name = 'sheet1', index = False)

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

        axs[1,3].plot(framenumber, wrist_angle_data)
        axs[1,3].set_title('wrist angle')

        axs[2,3].plot(framenumber, wrist_angular_vel_smooth)
        axs[2,3].set_title('wrist angular velocity')

        """
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
