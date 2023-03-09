import streamlit as st
import cv2 as cv
import face_recognition
import pickle
import os

path = os.getcwd()
encoded_dir = str(path) + "/dossier_encoded/"

# Charger les listes depuis le fichier pickle
with open(str(encoded_dir)+'known_faces.pkl', 'rb') as f:
    known_face_encodings, known_face_names = pickle.load(f)


genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
Model_mean_values = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Homme', 'Femme']
genderNet = cv.dnn.readNet(genderProto,genderModel)

def detect_faces(frame, known_face_encodings, known_face_names):
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame, model='hog', number_of_times_to_upsample=1)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_names = []
    face_gender=[]

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        gender = "Non-binary"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = face_recognition.face_locations(gray)
            for x in faces:
                face=frame[max(x[0]-20,0):min(x[2]+20,frame.shape[0]),max(0,x[3]-20):min(x[1]+20,frame.shape[1])]
                genderNet.setInput(cv.dnn.blobFromImage(face, 1.0, (227, 227), Model_mean_values, swapRB=False))
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]
                
        face_gender.append(gender)
        face_names.append(name)
        

    for (top, right, bottom, left), name, gender in zip(face_locations, face_names, face_gender):
        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv.FILLED)
        font = cv.FONT_HERSHEY_DUPLEX
        label = name + " - " + gender
        cv.putText(frame, label, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)


if st.button('Open Camera'):
    st.write('Camera is open')
    video_stream = cv.VideoCapture(0)
    video_placeholder = st.empty()
    stop_button = st.button('Stop Camera')
    while not stop_button:
        ret, frame = video_stream.read()
        if ret:
            detect_faces(frame, known_face_encodings, known_face_names)
            video_placeholder.image(frame, channels="BGR")
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_stream.release()
    cv.destroyAllWindows()
    st.write('Camera is stopped')