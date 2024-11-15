import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

model = load_model('my_asl_model.h5')

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            img_array = cv2.resize(image_rgb, (64, 64))
            img_array = img_array / 255.0  
            img_array = np.expand_dims(img_array, axis=0)  

            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)

            if predicted_class.size > 0 and predicted_class[0] < len(classes):
                predicted_label = classes[predicted_class[0]]
            else:
                predicted_label = "Unknown"

            cv2.putText(frame, f'Predicted: {predicted_label}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('ASL Translator', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
