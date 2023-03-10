import streamlit as st
import cv2 as cv
import face_recognition
import pickle
import os
import numpy as np
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from PIL import Image

path = os.getcwd()
encoded_dir = str(path) + "/dossier_encoded/"
model_dir = str(path) + "/models/"
path_dir = str(path) + "/assets/"

# Charger les listes depuis le fichier pickle
with open(str(encoded_dir)+'known_faces.pkl', 'rb') as f:
    known_face_encodings, known_face_names = pickle.load(f)


#Model emotion
model_emotion = load_model(model_dir+'Emotion_Detection.h5', compile=False)

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
emotionsList = ['angry', 'happy', 'sad', 'surprise', 'neutral']
genderList = ['Male', 'Female']

def plot_age_indicator(face_age):
    # Valeur prédite de l'âge

    data_tuple = tuple(map(int, face_age[0].strip("() ").split(",")))

    # Calcul de la moyenne
    age_pred = sum(data_tuple) / len(data_tuple) 

    # Créer le graphique avec la jauge et le nombre
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = age_pred, 
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Age"},
        gauge = {
            'axis': {'range': [1, 100]},
            'bar': {'color': "green"},
        }
    ))
    fig.update_layout(width=280, height=280)
    return fig

def plot_emotion_wheel(emotion_scores):
    # Trier les scores par ordre décroissant pour obtenir les émotions les plus probables
    sorted_scores = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)

    # Définir les couleurs pour chaque quadrant de la Roue des émotions
    colors = {'Positive': '#00a8cc', 'Neutral': '#6c757d', 'Negative': '#ff6b6b'}

    # Définir les émotions pour chaque quadrant de la Roue des émotions
    emotions = {'Positive': ['happy', 'surprise'], 'Neutral': ['neutral'], 'Negative': ['angry', 'sad']}

    # Récupérer l'émotion la plus probable
    most_likely_emotion = sorted_scores[0][0]

    # Déterminer le quadrant de la Roue des émotions correspondant à l'émotion la plus probable
    for quadrant, emotions_list in emotions.items():
        if most_likely_emotion in emotions_list:
            break

    # Placer l'émotion sur la Roue des émotions en fonction de son intensité
    fig = go.Figure()
    fig.add_trace(go.Pie(values=[emotion_scores[most_likely_emotion], 1-emotion_scores[most_likely_emotion]], 
                        hole=.7, 
                        marker_colors=[colors[quadrant], '#f2f2f2'],
                        textinfo='none',
                        direction='clockwise',
                        rotation=0, 
                        labels=['Emotion', 'Non emotion']))
    fig.update_traces(hoverinfo='none', textfont_size=18)
    fig.update_layout(
        annotations=[
            dict(
                text=most_likely_emotion.capitalize(),
                font=dict(size=32, color=colors[quadrant]),
                showarrow=False,
                x=0.5,
                y=0.5
            )
        ],
        width=280,
        height=280,
        margin=dict(l=0, r=1, t=0, b=0),
        paper_bgcolor=None
    )
    return fig


def show_infos(frame, face_locations, face_names, face_gender, face_age, face_emotions):
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

def detect_faces(frame, known_face_encodings, known_face_names):
    #small_frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    
    face_locations = face_recognition.face_locations(rgb_frame, model='hog', number_of_times_to_upsample=1)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    face_gender=[]
    face_age=[]
    face_emotions =[]
    emotion_scores = {}
    un = False

    # Loop through each face encoding in the current frame of video
    for face_encoding in face_encodings:
        # See if the face is a match for any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        emotion = "Unemotional" 
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
        emotion = emotionsList[emotion_index]
        angry_score, happy_score, sad_score, surprise_score, neutral_score = emotion_predictions[0]

        # Placer les scores de prédiction dans un dictionnaire
        emotion_scores = {'angry': angry_score, 'happy': happy_score, 'sad': sad_score, 'surprise': surprise_score, 'neutral': neutral_score}
        # Add the name, gender, age, and emotion to the list of face names
        face_names.append(name)
        face_gender.append(gender)
        face_age.append(age)
        face_emotions.append(emotion)
    return face_locations, face_names, face_gender, face_age, face_emotions, emotion_scores

fig = go.Figure()   
if st.button('Open Camera'):
    st.write('Camera is open')
    # Open camera with OpenCV and keep in video stream:
    video_stream = cv.VideoCapture(0)
    video_stream.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    video_stream.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    frame_counter = 0
    video_placeholder = st.empty()

    # Create two columns
    col1, col2, col3  = st.columns([1, 1, 1])

    # Place the first chart in the first column
    with col1:
        graphe_emotion_placeholder = st.empty()
        
    # Place the second chart in the second column, on a new row
    with col2:
        graphe_age_placeholder = st.empty()

    with col3 :
        graphe_gender_placeholder = st.empty()

    stop_button = st.button('Stop Camera')
    while not stop_button:
        ret, frame = video_stream.read()
        if not ret:
            break
            
        if frame_counter == 0:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # Appliquer un filtre de flou pour réduire le bruit
            blurred = cv.GaussianBlur(gray, (5, 5), 0)
            # Appliquer un seuillage adaptatif pour améliorer le contraste
            threshold = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 4)

        frame_counter += 1
        print(frame_counter)
        if frame_counter == 4:
            frame_counter = 0

            face_locations, face_names, face_gender, face_age, face_emotions, emotion_scores = detect_faces(frame, known_face_encodings, known_face_names)
            show_infos(frame, face_locations, face_names, face_gender, face_age, face_emotions)

            #ca affiche 0 c'est bizarre 
            if len(face_age)==1: 
                fig_age = plot_age_indicator(face_age)
                graphe_age_placeholder.plotly_chart(fig_age, use_container_width=False)
            else :
                image = Image.open(path_dir+"age.png")
                max_size = (300, 300)
                image.thumbnail(max_size)
                graphe_age_placeholder.image(image)

            if len(face_emotions)>0: 
                fig_maj = plot_emotion_wheel(emotion_scores)
                graphe_emotion_placeholder.plotly_chart(fig_maj, use_container_width=False)
            elif ((len(face_emotions)==0) or (len(face_age)!=1)):
                image = Image.open(path_dir+"emotion.png")
                max_size = (300, 300)
                image.thumbnail(max_size)
                graphe_emotion_placeholder.image(image)

            if len(face_gender)==1 :
                if face_gender[0] == "Male":
                    men = Image.open(path_dir+"men.png")
                    max_size = (200, 200)
                    men.thumbnail(max_size)
                    graphe_gender_placeholder.image(men)
                elif face_gender[0] == "Female" :
                    wem = Image.open(path_dir+"wemen.png")
                    max_size = (200, 200)
                    wem.thumbnail(max_size)
                    graphe_gender_placeholder.image(wem)
            else :
                nonb = Image.open(path_dir+"non-binary.png")
                max_size = (200, 200)
                nonb.thumbnail(max_size)
                graphe_gender_placeholder.image(nonb)

            # Display the resulting image            
            video_placeholder.image(frame, channels="BGR")

        # if cv.waitKey(1) & 0xFF == ord('q'):
        #     break

    
    video_stream.release()
    cv.destroyAllWindows()
    st.write('Camera is stopped')
