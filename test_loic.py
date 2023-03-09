import streamlit as st
import cv2 as cv
import face_recognition
import pickle
import os
import numpy as np
#from keras.preprocessing.image import img_to_array
#from tensorflow import keras
#from keras.models import load_model
from tensorflow.keras.models import load_model

path = os.getcwd()
encoded_dir = str(path) + "/dossier_encoded/"
model_dir = str(path) + "/models/"
enregistrement_dir = str(path) + "/enregistrement/"

# Charger les listes depuis le fichier pickle
with open(str(encoded_dir)+'known_faces.pkl', 'rb') as f:
    known_face_encodings, known_face_names = pickle.load(f)


#Model emotion
model_emotion = load_model(model_dir+'Emotion_Detection.h5', compile=False)
class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

#Model age
AGE_PROTO = 'age_deploy.prototxt'
AGE_MODEL = 'age_net.caffemodel'
age_net = cv.dnn.readNetFromCaffe(model_dir+AGE_PROTO,model_dir+AGE_MODEL)

AGE_INTERVALS = ['(0, 2)', '(4, 6)','(25, 32)', '(8, 12)', '(15, 20)',
                  '(38, 43)', '(48, 53)', '(60, 100)']

#Model gender
GENDER_PROTO = 'gender_deploy.prototxt'
GENDER_MODEL = 'gender_net.caffemodel'
gender_net = cv.dnn.readNetFromCaffe(model_dir+GENDER_PROTO, model_dir+GENDER_MODEL)


# Définir les émotions à prédire
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
genderList = ['Male', 'Female']

def detect_faces(frame, known_face_encodings, known_face_names):
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame, model='hog', number_of_times_to_upsample=1)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    face_gender=[]
    face_age=[]
    face_emotions =[]

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

        # Predict the gender of the face
        blob = cv.dnn.blobFromImage(face_img, scalefactor=1, size=(227, 227),mean=(78.4263377603, 87.7689143744, 114.895847746))
        
        # Determine the gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = genderList[gender_preds[0].argmax()]

        # Determine the age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = AGE_INTERVALS[age_preds[0].argmax()]

        #emotions
        face_img = cv.cvtColor(face_img, cv.COLOR_BGR2GRAY)
        face_img = cv.resize(face_img, (48, 48))
        face_img = face_img / 255.0
        face_img = face_img.reshape(1, 48, 48, 1)
        emotion_predictions = model_emotion.predict(face_img)
        # Trouver l'index de la prédiction d'émotion la plus élevée
        emotion_index = np.argmax(emotion_predictions)
        # Trouver le nom de l'émotion correspondant à l'index
        emotion = class_labels[emotion_index]

        # Add the name, gender, age, and emotion to the list of face names
        face_names.append(name)
        face_gender.append(gender)
        face_age.append(age)
        face_emotions.append(emotion)

    # Draw a rectangle around the face
    for (top, right, bottom, left), name, gender, age, emotion in zip(face_locations, face_names, face_gender, face_age, face_emotions):
        # Draw a purple box around the face
        cv.rectangle(frame, (left, top), (right, bottom), (238, 130, 238), 2)

        # Draw labels with the name, gender, age, and emotion below the face
        cv.rectangle(frame, (left, bottom), (right, bottom + 90), (238, 130, 238), cv.FILLED)
        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(frame, name, (left + 6, bottom + 20), font, 0.7, (255, 255, 255), 1)
        cv.putText(frame, gender, (left + 6, bottom + 40), font, 0.7, (255, 255, 255), 1)
        cv.putText(frame, age, (left + 6, bottom + 60), font, 0.7, (255, 255, 255), 1)
        cv.putText(frame, emotion, (left + 6, bottom + 80), font, 0.7, (255, 255, 255), 1)

    return frame





if st.button('Open Camera'):
    st.write('Camera is open')
    # Open camera with OpenCV and keep in video stream:
    video_stream = cv.VideoCapture(0)
    width = int(video_stream.get(cv.CAP_PROP_FRAME_WIDTH))
    heigth= int(video_stream.get(cv.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(str(enregistrement_dir)+'output_0.mp4',fourcc, 3.7, (width,heigth))
    video_placeholder = st.empty()
    stop_button = st.button('Stop Camera')
    while not stop_button:
        ret, frame = video_stream.read()
        if ret:
            frame = cv.flip(frame, 1)
            # Detect faces in the current frame of video
            detect_faces(frame, known_face_encodings, known_face_names)
            # Write the frame to the output file
            out.write(frame)

            # Display the resulting image
            video_placeholder.image(frame, channels="BGR")
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_stream.release()
    out.release()
    cv.destroyAllWindows()
    st.write('Camera is stopped')
    
    video_stream.release()
    cv.destroyAllWindows()
    st.write('Camera is stopped')