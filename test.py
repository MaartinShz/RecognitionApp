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

def detect_faces(frame, known_face_encodings, known_face_names):
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_names = []

    # Loop through each face encoding in the current frame of video
    for face_encoding in face_encodings:
        # See if the face is a match for any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerence=0.5)
        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)

    # Draw a box around each face and label with name
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw a box around the face
        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv.FILLED)
        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)


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
