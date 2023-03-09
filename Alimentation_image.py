import streamlit as st
import cv2 as cv
import face_recognition
import pickle
import os


path = os.getcwd()
encoded_dir = str(path) + "/dossier_encoded/"
test = "C:/Users/spica/Downloads"



st.write(" Ou alors prenez une photo avec la webcam :")


# Lancement de la webcam
user_input = st.text_input("Entrez votre prénom et nom si vous souhaitez prendre une photo : ")
if st.button("Take a photo with webcam"):
    if len(user_input) == 0:
        st.warning("Veuillez entrer votre prénom et nom !")
    else :
        cam = cv.VideoCapture(0)
        ret, frame = cam.read()
        cv.imwrite(str(test)+ "/" + str(user_input) + ".jpeg", frame)


    
    
# Sélection de l'image sur streamlit
image_file = st.file_uploader("Sélectionner une image", type='jpeg')

# Charger les listes depuis le fichier pickle
with open(str(encoded_dir)+'known_faces.pkl', 'rb') as f:
    known_face_encodings, known_face_names = pickle.load(f)

if image_file is not None:
    image = face_recognition.load_image_file(image_file)
    face_encoding = face_recognition.face_encodings(image)[0]
    st.write(image_file.name.split(".")[0])
    known_face_encodings.append(face_encoding)
    known_face_names.append(image_file.name.split(".")[0])
    st.write(known_face_names)