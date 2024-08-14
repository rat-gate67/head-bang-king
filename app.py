import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import dlib
from imutils import face_utils

import threading

PREDICTOR_PATH = "data/shape_predictor_68_face_landmarks.dat"
DETECTOR = dlib.get_frontal_face_detector() # 顔検出
PREDICTOR = dlib.shape_predictor(PREDICTOR_PATH) # 顔ランドマーク検出


# Session State Initialization
if 'count' not in st.session_state:
    st.session_state['count'] = 0
if 'up' not in st.session_state:
    st.session_state['up'] = True
if 'down' not in st.session_state:
    st.session_state['down'] = False

st.title("My first Streamlit app")
st.write("Hello, world")


lock = threading.Lock()

        
def update_session_state(pitch):
    with lock:
        if st.session_state["up"] and pitch > 25:
            st.session_state["up"] = False
            st.session_state["down"] = True
            st.session_state["count"] += 1
        elif st.session_state["down"] and pitch < -25:
            st.session_state["up"] = True
            st.session_state["down"] = False

def callback(frame):
    img = frame.to_ndarray(format="bgr24")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # グレースケール変換
    rects = DETECTOR(gray, 0) # 顔検出



    for rect in rects:
        shape = PREDICTOR(gray, rect)
        shape = face_utils.shape_to_np(shape)
        for (x, y) in shape:
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
        
        image_points = np.array([
                tuple(shape[30]),#鼻頭
                tuple(shape[21]),
                tuple(shape[22]),
                tuple(shape[39]),
                tuple(shape[42]),
                tuple(shape[31]),
                tuple(shape[35]),
                tuple(shape[48]),
                tuple(shape[54]),
                tuple(shape[57]),
                tuple(shape[8]),
                ],dtype='double')
    
    if len(rects) > 0:
        
        model_points = np.array([
            (0.0,0.0,0.0), # 30
            (-30.0,-125.0,-30.0), # 21
            (30.0,-125.0,-30.0), # 22
            (-60.0,-70.0,-60.0), # 39
            (60.0,-70.0,-60.0), # 42
            (-40.0,40.0,-50.0), # 31
            (40.0,40.0,-50.0), # 35
            (-70.0,130.0,-100.0), # 48
            (70.0,130.0,-100.0), # 54
            (0.0,158.0,-10.0), # 57
            (0.0,250.0,-50.0) # 8
            ])

        size = img.shape
        print(size)

        focal_length = size[1]
        center = (size[1]//2, size[0]//2)

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],], dtype = "double")
        
        dist_coeffs = np.zeros((4,1))

        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        #回転行列とヤコビアン
        (rotation_matrix, jacobian) = cv2.Rodrigues(rotation_vector)
        mat = np.hstack((rotation_matrix, translation_vector))

        #yaw,pitch,rollの取り出し
        (_, _, _, _, _, _, eulerAngles) = cv2.decomposeProjectionMatrix(mat)
        yaw = eulerAngles[1]
        pitch = eulerAngles[0]
        roll = eulerAngles[2]

        # update_session_state(pitch=pitch)
        
        cv2.putText(img, f"yaw={yaw}", (20, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.putText(img, f"pitch={pitch}", (20, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.putText(img, f"roll={roll}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        # cv2.putText(img, f"count={st.session_state['count']}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rotation_vector,translation_vector, camera_matrix, dist_coeffs)
        #計算に使用した点のプロット/顔方向のベクトルの表示
        for p in image_points:
            cv2.drawMarker(img, (int(p[0]), int(p[1])),  (0.0, 1.409845, 255),markerType=cv2.MARKER_CROSS, thickness=1)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv2.arrowedLine(img, p1, p2, (255, 0, 0), 2)

    # img = cv2.cvtColor(cv2.Canny(img, threshold1, threshold2), cv2.COLOR_GRAY2BGR)



    return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="example",
    video_frame_callback=callback,
    rtc_configuration={  # この設定を足す
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)