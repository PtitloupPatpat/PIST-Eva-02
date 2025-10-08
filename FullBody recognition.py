### 08/10/2025 PIST SMR Supmicrotech JOACHIM Tom ###

import mediapipe
import cv2
import time

video_capture = cv2.VideoCapture(0)

pose_capture = mediapipe.solutions.pose
tracerLigne = mediapipe.solutions.drawing_utils

pose = pose_capture.Pose()

PreviousTime = 0

while(True):
    ret, frame = video_capture.read()                       # On lit une image (frame) de la vidéo
    imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)       # On convertie l’image qui est en BGR en RGB

    result = pose.process(imageRGB)                         # Détecte le corps et récupère ces coordonnés

    if (result.pose_landmarks):                             # Vérifie si le résultat ne pas nulle

        ### Traçage des lignes ###

        for landmark in result.pose_landmarks.landmark:
            hauteur, largeur, longeur = frame.shape
            positionX, positionY, positionZ = (int(landmark.x * largeur), int(landmark.y * hauteur),
                                               int(landmark.z * longeur))


        ### Dessine un grand cercle ###


            cv2.circle(frame, (positionX, positionY), 25, (255, 0, 255), cv2.FILLED)
            tracerLigne.draw_landmarks(frame, result.pose_landmarks, pose_capture.POSE_CONNECTIONS)

    CurrentTime = time.time()
    fps = 1 / (CurrentTime - PreviousTime)
    PreviousTime = CurrentTime
    cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)

    cv2.imshow("Frame",frame)

    if cv2.waitKey(20) & 0xFF == ord('d'):
        break

video_capture.release()
cv2.destroyAllWindows()