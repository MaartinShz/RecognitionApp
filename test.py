import streamlit as st
import cv2 as cv
import face_recognition
import pickle
import os
import numpy as np
#from keras_vggface.vggface import VGGFace
#from keras_vggface.utils import preprocess_input

path = os.getcwd()
encoded_dir = str(path) + "/dossier_encoded/"

# Charger les listes depuis le fichier pickle
with open(str(encoded_dir)+'known_faces.pkl', 'rb') as f:
    known_face_encodings, known_face_names = pickle.load(f)

# Charger le modèle VGGFace pré-entraîné sur la base de données Fer2013 pour la reconnaissance d'émotions
#emotion_model = VGGFace(model='resnet50')

def predict_emotion(frame, face_locations):
    # Récupérer le visage détecté et le recadrer pour avoir uniquement le visage
    (top, right, bottom, left) = face_locations[0]
    face = frame[top:bottom, left:right]
    # Redimensionner l'image du visage pour l'adapter à l'entrée du modèle
    face = cv.resize(face, (224, 224))
    # Prétraiter l'image du visage pour l'adapter à l'entrée du modèle VGGFace
    face = preprocess_input(face.astype(np.float32), version=2)
    # Prédire l'émotion à partir de l'image du visage en utilisant le modèle VGGFace
    preds = emotion_model.predict(np.expand_dims(face, axis=0))[0]
    # Récupérer l'émotion avec le score le plus élevé
    emotion = emotions[np.argmax(preds)]
    return emotion

# Définir les émotions à prédire
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def detect_faces(frame, known_face_encodings, known_face_names):
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame, model='hog', number_of_times_to_upsample=1)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_names = []
    face_emotions = [] # Liste des émotions détectées pour chaque visage détecté

    # Loop through each face encoding in the current frame of video
    for face_encoding in face_encodings:
        # See if the face is a match for any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        emotion = "Unknown" # Par défaut, l'émotion est inconnue

        # If a match was found in known_face_encodings, just use the first one
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Predict emotion for the detected face
        #if len(face_locations) > 0:
            #emotion = predict_emotion(frame, face_locations)

        face_names.append(name)
        #face_emotions.append(emotion)

    # Draw a box around each face and label with name and emotion
    for (top, right, bottom, left), name, emotion in zip(face_locations, face_names): #face_emotions
        # Draw a box around the face
        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name and emotion below the face
        label = name + " - " + emotion
        cv.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv.FILLED)
        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(frame, label, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)


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
