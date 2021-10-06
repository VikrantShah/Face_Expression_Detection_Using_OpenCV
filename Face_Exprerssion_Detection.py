# pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
# pip install facial-emotion-recognition

from facial_emotion_recognition import EmotionRecognition
import cv2

er = EmotionRecognition(device = "gpu")

cam = cv2.VideoCapture(0)

while True :
    success, frame = cam.read()
    
    frame = er.recognise_emotion(frame, return_type = "BGR")
    
    cv2.imshow("Emotion Recognition", frame)
    
    key = cv2.waitKey(1)
    if key == 27 :
        break
    
cam.release()
cv2.destroyAllWindows()