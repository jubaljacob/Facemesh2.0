import cv2
import mediapipe as mp
import time


class faceDetector():

    def __init__(self, mode=False, max_faces=4, refine=False, detectionConf=0.5, trackingConf=0.5):

        self.mode = mode
        self.max_faces = max_faces
        self.refine = refine
        self.detectionConf = detectionConf
        self.trackingConf = trackingConf

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            self.mode, self.max_faces, self.detectionConf, self.trackingConf)

        self.drawSpec = self.mpDraw.DrawingSpec(
            color=[0, 255, 0], thickness=1, circle_radius=1)

    def findFace(self, frame, draw=True):

        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.result = self.faceMesh.process(frameRGB)

        if self.result.multi_face_landmarks:
            for faceLms in self.result.multi_face_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(
                        frame, faceLms, self.mpFaceMesh.FACEMESH_FACE_OVAL, self.drawSpec, self.drawSpec)

                # for id, lm in faceLms.landmark:
                #     h, w, c = frame.shape
                #     cx, cy = int(lm.x * w), int(lm.y * h)
                #     print(id ,cx, cy)
        return frame


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0
    detector = faceDetector()
    while True:
        success, frame = cap.read()
        frame = detector.findFace(frame)

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


if __name__ == "__main__":
    main()
