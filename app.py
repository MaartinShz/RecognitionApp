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
                text = r.recognize_google(audio, language='fr-FR' ,show_all=True)# change language #'en-US' # ou 'en-GB'
                
            
                listtext = []
                if(len(text)>0):   
                    for i in text["alternative"]:
                        listtext.append(i["transcript"])

                 #return listetext

            except sr.UnknownValueError:
                text = "Je n'ai pas compris ce que vous avez dit"
                 #st.write("Je n'ai pas compris ce que vous avez dit")
            except sr.RequestError as e:
                text="Une erreur s'est produite : {}".format(e)
                 #st.write("Une erreur s'est produite : {}".format(e))
            return listtext             
     
    while True:
        listtext = recognize_speech()
        st.write(listtext)
        
        for wordlisttext in listtext:
            
            # if("démarre webcam" in wordlisttext):
            #     st.warning("Start WebCAM !!!!!!!!!!!!!!!!")
            #     break
            # elif("arrête webcam" in wordlisttext):
            #     st.warning("Stop WebCAM !!!!!!!!!!!!!!!!")
            #     break
            # elif("démarre reconnaissance" in wordlisttext):
            #     st.warning("Start detection !!!!!!!!!!!!!!!!")
            #     break
            # elif("arrête reconnaissance" in wordlisttext):
            #     st.warning("Stop detection !!!!!!!!!!!!!!!!")
            #     break
            # elif("démarre enregistrement" in wordlisttext):
            #     st.warning("Start recording !!!!!!!!!!!!!!!!")
            #     break
            # elif("arrête enregistrement" in wordlisttext):
            #     st.warning("Stop recording !!!!!!!!!!!!!!!!")
            #     break
            
            ###
            
            if("démarre") in wordlisttext:
                st.warning("start")
                if("webcam") in wordlisttext:
                    st.warning("webcam")
                    break
                elif("reconnaissance") in wordlisttext:
                    st.warning("detection")
                    break
                elif("enregistrement") in wordlisttext:
                    st.warning("recording")
                    break
                
            elif("arrête") in wordlisttext:
                st.warning("stop")
                if("webcam") in wordlisttext:
                    st.warning("webcam")
                    break
                elif("reconnaissance") in wordlisttext:
                    st.warning("detection")
                    break
                elif("enregistrement") in wordlisttext:
                    st.warning("recording")
                    break
                
           
            
            
            


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