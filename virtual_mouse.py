import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# Screen dimensions
screen_w, screen_h = pyautogui.size()

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Variables for smoothing and gestures
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0
smoothening = 5
click_distance_threshold = 0.05
scroll_mode = False
drag_mode = False
right_click_mode = False
double_click_threshold = 0.3  # seconds
last_click_time = 0
frame_count = 0

# For cursor speed control (based on hand size)
base_hand_size = 0.1  # Reference hand size
current_hand_size = 0

# For scroll control
scroll_start_y = 0
scroll_sensitivity = 50

def fingers_extended(landmarks):
    """Check which fingers are extended"""
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    extended = [False, False, False, False]
    
    # Check each finger (except thumb)
    for i, tip_id in enumerate(finger_tips):
        # Finger is extended if tip is higher than middle joint
        if landmarks[tip_id].y < landmarks[tip_id-2].y:
            extended[i] = True
    
    return extended

while True:
    success, img = cap.read()
    if not success:
        continue
        
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            
            # Get hand size for cursor speed control (wrist to middle finger)
            current_hand_size = np.sqrt(
                (landmarks[0].x - landmarks[9].x)**2 + 
                (landmarks[0].y - landmarks[9].y)**2
            )
            
            # Check which fingers are extended
            extended = fingers_extended(landmarks)
            index_ext, middle_ext, ring_ext, pinky_ext = extended
            
            # Get index finger tip (landmark 8)
            index_tip = landmarks[8]
            thumb_tip = landmarks[4]
            
            # Convert coordinates to screen size
            curr_x = int(index_tip.x * screen_w)
            curr_y = int(index_tip.y * screen_h)
            
            # Adjust cursor speed based on hand size
            speed_factor = base_hand_size / current_hand_size if current_hand_size > 0 else 1
            speed_factor = max(0.5, min(2.0, speed_factor))  # Limit the speed factor
            
            # Smoothing movement with speed factor
            smooth_x = prev_x + (curr_x - prev_x) / (smoothening / speed_factor)
            smooth_y = prev_y + (curr_y - prev_y) / (smoothening / speed_factor)
            
            # Calculate distance between thumb and index for click
            distance = np.sqrt((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2)
            
            # SCROLL MODE (middle finger extended)
            if middle_ext and not any(extended[:1] + extended[2:]):
                if not scroll_mode:
                    scroll_mode = True
                    scroll_start_y = curr_y
                    cv2.putText(img, "Scroll Mode", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                
                # Calculate scroll amount based on vertical movement
                scroll_amount = (scroll_start_y - curr_y) / scroll_sensitivity
                pyautogui.scroll(int(scroll_amount))
                scroll_start_y = curr_y
            else:
                scroll_mode = False
                
                # REGULAR CURSOR MOVEMENT
                if not drag_mode:
                    pyautogui.moveTo(smooth_x, smooth_y)
                prev_x, prev_y = smooth_x, smooth_y
            
            # RIGHT CLICK (pinky extended, others not)
            if pinky_ext and not any(extended[:3]):
                if not right_click_mode:
                    pyautogui.rightClick()
                    right_click_mode = True
                    cv2.putText(img, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                right_click_mode = False
                
                # LEFT CLICK (thumb and index close)
                if distance < click_distance_threshold:
                    current_time = time.time()
                    time_since_last_click = current_time - last_click_time
                    
                    # DOUBLE CLICK (two quick taps)
                    if time_since_last_click < double_click_threshold:
                        pyautogui.doubleClick()
                        cv2.putText(img, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
                        last_click_time = 0  # Prevent triple clicks
                    else:
                        # DRAG MODE (index and middle fingers close)
                        if middle_ext and distance < click_distance_threshold * 1.2:
                            if not drag_mode:
                                pyautogui.mouseDown()
                                drag_mode = True
                                cv2.putText(img, "Drag Mode", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
                        else:
                            # SINGLE CLICK
                            pyautogui.click()
                            cv2.putText(img, "Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                            last_click_time = current_time
                else:
                    if drag_mode:
                        pyautogui.mouseUp()
                        drag_mode = False
            
            # Display hand size info for debugging
            cv2.putText(img, f"Speed: {speed_factor:.1f}x", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            # Draw hand landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Display mode information
    if scroll_mode:
        mode_text = "SCROLL MODE"
        color = (0, 255, 255)  # Yellow
    elif drag_mode:
        mode_text = "DRAG MODE"
        color = (255, 0, 255)  # Purple
    elif right_click_mode:
        mode_text = "RIGHT CLICK"
        color = (0, 0, 255)    # Red
    else:
        mode_text = "NORMAL MODE"
        color = (0, 255, 0)    # Green
    
    cv2.putText(img, mode_text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow("Advanced Virtual Mouse", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()