import streamlit as st
import cv2 as cv
import face_recognition
import pickle
import os
import numpy as np
from keras.models import load_model


path = os.getcwd()
encoded_dir = str(path) + "/dossier_encoded/"
model_dir = str(path) + "/models/"

# Charger les listes depuis le fichier pickle
with open(str(encoded_dir)+'known_faces.pkl', 'rb') as f:
    known_face_encodings, known_face_names = pickle.load(f)


#ace_classifier = cv.CascadeClassifier('./haarcascade_frontalface_default.xml')
#classifier =load_model('./Emotion_Detection.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']
#__________


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_INTERVALS = ['(0, 2)', '(4, 6)','(25, 32)', '(8, 12)', '(15, 20)',
                  '(38, 43)', '(48, 53)', '(60, 100)']

genderList = ['Male', 'Female']

#Model age
AGE_PROTO = 'age_deploy.prototxt'
AGE_MODEL = 'age_net.caffemodel'
age_net = cv.dnn.readNetFromCaffe(model_dir+AGE_PROTO,model_dir+AGE_MODEL)
#__________

#Model gender
GENDER_PROTO = 'gender_deploy.prototxt'
GENDER_MODEL = 'gender_net.caffemodel'
gender_net = cv.dnn.readNetFromCaffe(model_dir+GENDER_PROTO, model_dir+GENDER_MODEL)
#__________

# def predict_emotion(frame, face_locations):
#     # Récupérer le visage détecté et le recadrer pour avoir uniquement le visage
#     (top, right, bottom, left) = face_locations[0]
#     face = frame[top:bottom, left:right]
#     # Redimensionner l'image du visage pour l'adapter à l'entrée du modèle
#     face = cv.resize(face, (224, 224))
#     # Prétraiter l'image du visage pour l'adapter à l'entrée du modèle VGGFace
#     face = preprocess_input(face.astype(np.float32), version=2)
#     # Prédire l'émotion à partir de l'image du visage en utilisant le modèle VGGFace
#     preds = emotion_model.predict(np.expand_dims(face, axis=0))[0]
#     # Récupérer l'émotion avec le score le plus élevé
#     emotion = emotions[np.argmax(preds)]
#     return emotion

# Définir les émotions à prédire
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def detect_faces(frame, known_face_encodings, known_face_names):
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame, model='hog', number_of_times_to_upsample=1)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_names = []
    face_emotions = [] 
    face_gender=[]
    face_age=[]

    # Loop through each face encoding in the current frame of video
    for face_encoding in face_encodings:
        # See if the face is a match for any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        emotion = "Unknown" 
        gender = "Non-binary"
        age = "Older"

        # If a match was found in known_face_encodings, just use the first one
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Predict gender and age for the detected face
        (top, right, bottom, left) = face_locations[0]
        face_img = frame[top:bottom, left:right]
        blob = cv.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                     MODEL_MEAN_VALUES,
                                     swapRB=False)
        # Predict Age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        i = age_preds[0].argmax()
        age = AGE_INTERVALS[i]
        
        # Predict Gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = genderList[gender_preds[0].argmax()]

        face_names.append(name)
        face_gender.append(gender)
        face_age.append(age)

    # Draw a box around each face and label with name, gender and age
    for (top, right, bottom, left), name, gender, age in zip(face_locations, face_names, face_gender, face_age):
        # Draw a box around the face
        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with the name, gender and age below the face
        label = f"{name}, {gender}, {age}"
        cv.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv.FILLED)
        cv.putText(frame, label, (left + 6, bottom - 6), cv.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    return frame


if st.button('Open Camera'):
    st.write('Camera is open')
    # Open camera with OpenCV and keep in video stream:
    video_stream = cv.VideoCapture(0)
    video_placeholder = st.empty()
    stop_button = st.button('Stop Camera')
    while not stop_button:
        ret, frame = video_stream.read()
        if ret:
            # Detect faces in the current frame of video
            detect_faces(frame, known_face_encodings, known_face_names)
            # Display the resulting image
            video_placeholder.image(frame, channels="BGR")
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_stream.release()
    cv.destroyAllWindows()
    st.write('Camera is stopped')