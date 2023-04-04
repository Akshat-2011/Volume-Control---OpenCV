import mediapipe as mp
import cv2

vid_cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
while True:
    success,img = vid_cap.read()
    flip_img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(flip_img,cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    #print(result.multi_hand_landmarks)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                h,w,c = flip_img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                print(id,cx,cy)
            mpDraw.draw_landmarks(flip_img,handLms,mpHands.HAND_CONNECTIONS)

    cv2.imshow('Image',flip_img)
    cv2.waitKey(1)

