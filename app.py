#DEBUG MODE #streamlit run --global.developmentMode=true app.py
import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px

import speech_recognition as sr



def home():
    st.title("Home 🏠")
    st.header("Welcome in Home page !")
    
    r = sr.Recognizer()
    
    def recognize_speech():
        with sr.Microphone() as source:
            st.write("Parlez...")
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.listen(source)
            try:
                text = r.recognize_google(audio, language='en-GB' ,show_all=True)# change language #'en-US' # ou 'en-GB'
                
            
                latlng = []
                if(len(text)>0):   
                    for i in text["alternative"]:
                        latlng.append(i["transcript"])

                 #return latlng

            except sr.UnknownValueError:
                text = "Je n'ai pas compris ce que vous avez dit"
                 #st.write("Je n'ai pas compris ce que vous avez dit")
            except sr.RequestError as e:
                text="Une erreur s'est produite : {}".format(e)
                 #st.write("Une erreur s'est produite : {}".format(e))
            return latlng             
     
    while True:
        latlng = recognize_speech()
        st.write(latlng)
        
        if("start webcam" in latlng):
            st.warning("Start WebCAM !!!!!!!!!!!!!!!!")
        elif("turn off webcam" in latlng):
            st.warning("Stop WebCAM !!!!!!!!!!!!!!!!")
            
        elif("start detection" in latlng):
            st.warning("Start detection !!!!!!!!!!!!!!!!")
        elif("turn off detection" in latlng):
            st.warning("Stop detection !!!!!!!!!!!!!!!!")
        
        elif("start recording" in latlng):
            st.warning("Start recording !!!!!!!!!!!!!!!!")
        elif("turn off recording" in latlng):
            st.warning("Stop recording !!!!!!!!!!!!!!!!")


################################################################################################################################################################################################################################################
def page1():
    st.title("Page 1 📺")
    
    
################################################################################################################################################################################################################################################
def page2():
    st.title("Page 2 🛡")
    
    
    
   
################################################################################################################################################################################################################################################
def page3():
    st.title("Page 3 ⚔")
    

    
    



    
 
    
    
##################################### MENU ################################################################################################################################################
# page list
pages = {
    "Home 🏠": home,
    "Page 1 📺": page1,
    "Page 2 🛡": page2,
    "Page 3 ⚔": page3
}

# Navigation bar
st.sidebar.title("Menu 📊")
selection = st.sidebar.radio("Aller à", list(pages.keys()))

# exexute page select
page = pages[selection]
page()