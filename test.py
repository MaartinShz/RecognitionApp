import face_recognition
import streamlit as st
import cv2 as cv

st.title("Face Recognition")
st.write('This is a simple face recognition app')

#st.button('Open Camera')

if st.button('Open Camera'):
    st.write('Camera is open')
    #open camera with opencv and keep in video stream:
    video_stream = cv.VideoCapture(0)
    while True:
        ret, frame = video_stream.read()
        if ret :
            cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    video_stream.release()
    cv.destroyAllWindows()



