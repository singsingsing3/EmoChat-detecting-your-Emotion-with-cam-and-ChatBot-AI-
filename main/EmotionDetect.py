import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

class EmotionDetector:
    def __init__(self):
        self.model = load_model('main\emotion_ResNet50.h5')
        self.class_names = ['anger', 'happy', 'normal', 'sad', 'worry']
        self.camera = cv.VideoCapture(0)
        
        self.emotion = "none"
        self.confidence = 0.0

        if not self.camera.isOpened():
            print("오류: 카메라를 열 수 없습니다.")
            exit()

    def predict_emotion(self):
        ret, frame = self.camera.read()
        if ret:
 

            frame_image = cv.resize(frame, (180, 180))
            frame_image = frame_image.astype("float32") / 255.0
            frame_image = np.expand_dims(frame_image, axis=0)

            try:
                proba = self.model.predict(frame_image)[0]
                idx = np.argmax(proba)
                self.emotion = self.class_names[idx]
                self.confidence = np.round(proba[idx] * 100, 2)
            except Exception as e:
                print(f"이미지 예측 중 오류 발생: {e}")

        return self.emotion, self.confidence

    def get_frame(self):
        ret, frame = self.camera.read()
        if ret:
            cv.putText(frame, f"{self.emotion} ({self.confidence:.2f}%)", (80, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            return frame
        else:
            return None
