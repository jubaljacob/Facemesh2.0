import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0
cTime = 0
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=3)
drawSpec = mpDraw.DrawingSpec(color=[0, 255, 0], thickness=1, circle_radius=1)


while True:
    success, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = faceMesh.process(frameRGB)

    if result.multi_face_landmarks:
        for faceLms in result.multi_face_landmarks:
            mpDraw.draw_landmarks(
                frame, faceLms, mpFaceMesh.FACEMESH_FACE_OVAL, drawSpec, drawSpec)
            for lm in faceLms.landmark:
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(cx, cy)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, str(int(fps)), (2, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break
print("code complete")
