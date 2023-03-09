import streamlit as st
import cv2 as cv
import face_recognition
import pickle
import os



def page3():
    st.title("Page 3 ⚔")
    path = os.getcwd()
    encoded_dir = str(path) + "/dossier_encoded/"
    etu_dir = str(path) + "/etu/"
    
    
    st.write(" Ou alors prenez une photo avec la webcam :")
    
    
    # Lancement de la webcam
    user_input = st.text_input("Entrez votre prénom et nom si vous souhaitez prendre une photo : ")
    if st.button("Take a photo with webcam"):
        if len(user_input) == 0:
            # On vérifie que l'utilisateur a bien entré son nom et prénom pour créer le nom de l'image
            st.warning("Veuillez entrer votre prénom et nom !")
        else :
            # On enregistre la photo dans le dossier etu
            cam = cv.VideoCapture(0)
            ret, frame = cam.read()
            cv.imwrite(str(etu_dir) + str(user_input) + ".jpeg", frame)
            st.warning("Photo Enregistrée dans le dossier /etu !")
       
    # Sélection de l'image sur streamlit
    image_file = st.file_uploader("Sélectionner une image", type='jpeg')
    
    # Charger les listes depuis le fichier pickle
    with open(str(encoded_dir)+'known_faces.pkl', 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)
    
    # Ajouter l'image et le nom dans les listes réspectives 
    if image_file is not None:
        image = face_recognition.load_image_file(image_file)
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(image_file.name.split(".")[0])
    
    
    # Enregistrer les listes dans un fichier pickle
    with open(str(encoded_dir)+'known_faces.pkl', 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)
