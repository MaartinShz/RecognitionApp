#DEBUG MODE #streamlit run --global.developmentMode=true app.py
#NORMAL MODE #streamlit run app.py

import streamlit as st #pip install streamlit

import plotly.express as px

import speech_recognition as sr #pip install SpeechRecognition
import cv2 as cv #pip install opencv-python
import face_recognition
import plotly.graph_objects as go
from PIL import Image
#import page Alimentation image
import pickle
import os

# import page Visage Detection
from keras.models import load_model

import webcam


path = os.getcwd()
encoded_dir = str(path) + "/dossier_encoded/"
model_dir = str(path) + "/models/"
enregistrement_dir = str(path) + "/enregistrement/"
path_dir = str(path) + "/assets/"

# Charger les listes depuis le fichier pickle
with open(str(encoded_dir)+'known_faces.pkl', 'rb') as f:
    known_face_encodings, known_face_names = pickle.load(f)



def acceuil():
    st.title("Reconnaissance faciale et d'émotions sur la webcam")
    st.header("Bienvenue sur la page de notre application !")
    st.write("Notre application propose plusieurs fonctionalités que vous pourrez retrouvez dans les différents onglets.")
    st.write("Vous pourrez utiliser notre application pour détecter les visages et les émotions des personnes présentes sur la webcam de votre ordinateur.")
    st.write("Vous pourrez également importer ou prendre une photo afin d'alimenter notre base d'apprentissage des visages.")

################################################################################################################################################################################################################################################

def Application():
    st.title("Comment utiliser notre application")
    st.write("Pour l'utiliser, cliquez sur le bouton 'Ouvrir la webcam' ci-dessous pour ouvrir la webcam de votre ordinateur et lancer l'enregistrement la vidéo.")
    st.write("Une fenêtre s'ouvrira et vous pourrez voir la webcam de votre ordinateur. La détection des visages et des émotions se fera en temps réel.")
    st.write("Vous pourrez également profiter de visualisations des émotions et des âges des personnes détectées sur la vidéo.")
    st.write("Stoppez l'affichage et l'enregistrement de la webcam en cliquant sur le bouton 'Stop webcam'.")
    st.write("Vous pourrez ensuite télécharger la vidéo en cliquant sur le bouton 'Télécharger la vidéo' (voir panneau latéral).")
    st.write("Nous souhaitions implémenter la reconnaissance vocale pour lancer l'enregistrement de la vidéo, mais nous n'avons pas réussi à le faire fonctionner pour toutes les fonctionnalitées.")
    st.write("Si vous cliquez sur le bouton 'Lancer la reconnaissance vocale', vous pourrez lancer l'enregistrement de la vidéo en disant ''Démarrer la webcam''.")
    open_webcam = st.button('Ouvrir la webcam')
    fig = go.Figure()
    def affichage_webcam():
        st.write("La webcam s'ouvre")
        # Open camera with OpenCV and keep in video stream:
        video_stream = cv.VideoCapture(0)
        width = int(video_stream.get(cv.CAP_PROP_FRAME_WIDTH))
        heigth= int(video_stream.get(cv.CAP_PROP_FRAME_HEIGHT))
        frame_counter = 0
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(str(enregistrement_dir)+'output_0.mp4',fourcc, 3.7, (width,heigth))
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

        stop_button = st.button('Arrêter la webcam')
        while not stop_button:
            ret, frame = video_stream.read()
            if ret:
                frame = cv.flip(frame, 1)
                face_locations, face_names, face_gender, face_age, face_emotions, emotion_scores = webcam.detect_faces(frame, known_face_encodings, known_face_names)
                webcam.show_infos(frame, face_locations, face_names, face_gender, face_age, face_emotions)
                out.write(frame)

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
                    face_locations, face_names, face_gender, face_age, face_emotions, emotion_scores = webcam.detect_faces(frame, known_face_encodings, known_face_names)
                    webcam.show_infos(frame, face_locations, face_names, face_gender, face_age, face_emotions)

                    #ca affiche 0 c'est bizarre 
                    if len(face_age)==1: 
                        fig_age = webcam.plot_age_indicator(face_age)
                        graphe_age_placeholder.plotly_chart(fig_age, use_container_width=False)
                    else :
                        image = Image.open(path_dir+"age.png")
                        max_size = (300, 300)
                        image.thumbnail(max_size)
                        graphe_age_placeholder.image(image)

                    if len(face_emotions)>0: 
                        fig_maj = webcam.plot_emotion_wheel(emotion_scores)
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
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        #if stop_button:
            #st.experimental_rerun()
        
        video_stream.release()
        out.release()
        cv.destroyAllWindows()
        st.write('Camera is stopped')
    if open_webcam:
        affichage_webcam()
    st.sidebar.title("Enregistrement de la vidéo")
    st.sidebar.header("Vous pouvez enregistrer la vidéo ou la supprimer en cliquant sur les boutons ci-dessous.")
    st.sidebar.write("Pensez bien à lancer la webcam et de l'éteindre avant de cliquer sur les boutons.")

    if st.sidebar.button('Supprimer la vidéo'):
        st.write("La vidéo va être supprimé...")
        os.remove(str(enregistrement_dir)+'output_0.mp4')
        st.experimental_rerun()

    if st.sidebar.button('Enregistrer la vidéo'):
        st.write("La vidéo va être enregistrée...")
        st.experimental_rerun()

    def recognize_speech():
            r = sr.Recognizer()
            with sr.Microphone() as source:
                st.write("Parlez...")
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio = r.listen(source)
                try:
                    text = r.recognize_google(audio, language='fr-FR' ,show_all=True)# change language #'en-US' # ou 'en-GB'
                    listtext = []
                    if(len(text)>0):   
                        for i in text["alternative"]:
                            listtext.append(i["transcript"])
                            
                except sr.UnknownValueError:
                    print("Je n'ai pas compris ce que vous avez dit")
                except sr.RequestError as e:
                    print("Une erreur s'est produite : {}".format(e))
        
                return listtext


    def reco_voc():
        listtext = recognize_speech()
        st.write(listtext)
        for wordlisttext in listtext:           
                
            if("démarre") in wordlisttext:
                #st.warning("start")
                if("webcam") in wordlisttext:
                        #st.warning("webcam")
                    affichage_webcam()
                    reco_voc()
                    break
                elif("reconnaissance") in wordlisttext:
                    st.warning("detection")
                    break
                elif("enregistrement") in wordlisttext:
                    st.warning("recording")
                    break
                    
            elif("arrête") in wordlisttext:
                #st.warning("stop")
                if("webcam") in wordlisttext:
                    stop_webcamp()
                    break
                elif("reconnaissance") in wordlisttext:
                    st.warning("detection")
                    break
                elif("enregistrement") in wordlisttext:
                    st.warning("recording")
                    break

    if st.button("Lancer la reconnaissance vocale"):
        reco_voc() 
   
################################################################################################################################################################################################################################################
def alimentation():
    st.title("Alimentation des Images")
    st.write("Vous pouvez ajouter des images dans le dossier /etu pour alimenter notre base.")
    st.write("Vous pouvez prendre une photo avec votre webcam qui sera importer en local dans le dossier /etu.")
    st.write("Il vous faudra ensuite importer la photo pour la transformer et l'ajouter à notre base.")

    path = os.getcwd()
    encoded_dir = str(path) + "/dossier_encoded/"
    etu_dir = str(path) + "/etu/"


    st.header(" Prendre une photo avec la Webcam :")


    # Open Webcam
    user_input = st.text_input("Entrez votre prénom et nom si vous souhaitez prendre une photo : ")
    if st.button("Take a photo with webcam"):
        if len(user_input) == 0:
            # check if user input his first and last name to create the image name
            st.warning("Veuillez entrer votre prénom et nom !")
        else :
            # registred photo in etu folder
            cam = cv.VideoCapture(0)
            ret, frame = cam.read()
            cv.imwrite(str(etu_dir) + str(user_input) + ".jpeg", frame)
            st.warning("Photo Enregistrée dans le dossier /etu !")
       
    # selection of the image on streamlit
    st.header(" Importer une photo :")
    image_file = st.file_uploader("Sélectionner une image", type='jpeg')

    # load list of images from pickle file
    with open(str(encoded_dir)+'known_faces.pkl', 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)

    # add image and name in each list 
    if image_file is not None:
        image = face_recognition.load_image_file(image_file)
        try : 
            face_encoding = face_recognition.face_encodings(image)[0]#
            known_face_encodings.append(face_encoding)
            known_face_names.append(image_file.name.split(".")[0])
            st.write("Photo bien enrégistrée dans la base !")
        except:
            st.warning("Attention Visage non détecté. Utilise une autre photo")
            os.remove(etu_dir + user_input + ".jpeg")


    # registred list in pickle file
    with open(str(encoded_dir)+'known_faces.pkl', 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)
    
################################################################################################################################################################################################################################################
def reconnaissance_vocale():
    st.title("Reconnaissance vocale")
    st.write("A l'origine nous souhaitions utiliser la reconnaissance vocale pour piloter notre application.")
    st.write("Malheureusement nous n'avons pas réussi à faire fonctionner la reconnaissance vocale pour toutes les fonctionnalitées.")
    st.write("Dans cet onglet vous pourrez webcamer les différentes fonctions vocales que nous avions mis en place.")
    st.write("Voici les différentes commandes que nous avons implémenté : ''Démarrer la webcam '',   ''Démarrer la détection '',    ''Démarrer l'enregistrement'',    ''Arrêter la webcam '',    '' Arrêter la détection ''   et   '' Arrêter l'enregistrement''")
    def recognize_speech():
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Parlez...")
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.listen(source)
            try:
                text = r.recognize_google(audio, language='fr-FR' ,show_all=True)# change language #'en-US' # ou 'en-GB'
                listtext = []
                if(len(text)>0):   
                    for i in text["alternative"]:
                        listtext.append(i["transcript"])
                        
            except sr.UnknownValueError:
                print("Je n'ai pas compris ce que vous avez dit")
            except sr.RequestError as e:
                print("Une erreur s'est produite : {}".format(e))
    
            return listtext             
     
    while True:
        listtext = recognize_speech()
        st.write(listtext)
        
        for wordlisttext in listtext:              
            if("démarre") in wordlisttext:
                #st.warning("start")
                if("webcam") in wordlisttext:
                    st.warning("Démmarage de la webcam")
                    break
                elif("reconnaissance") in wordlisttext:
                    st.warning("Démarrage de la detection")
                    break
                elif("enregistrement") in wordlisttext:
                    st.warning("Démmarage de l'enregistrement")
                    break
                
            elif("arrête") in wordlisttext:
                #st.warning("stop")
                if("webcam") in wordlisttext:
                    st.warning("Arrêt de la webcam")
                    break
                elif("reconnaissance") in wordlisttext:
                    st.warning("Arrêt de la detection")
                    break
                elif("enregistrement") in wordlisttext:
                    st.warning("Arrêt de l'enregistrement")
                    break
       
##################################### MENU ################################################################################################################################################
# page list
pages = {
    "Acceuil": acceuil,
    "Détection de visage et d'émotions": Application,
    "Alimentation des Images": alimentation,
    "Reconnaissance vocale": reconnaissance_vocale
}

# Navigation bar
st.sidebar.title("Menu 📊")
selection = st.sidebar.radio("Aller à", list(pages.keys()))

# Display Select Page
page = pages[selection]
page()