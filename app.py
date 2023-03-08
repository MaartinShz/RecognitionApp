# streamlit run --debug app.py
import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px

import speech_recognition as sr



def home():
    st.title("Accueil 🏠")
    st.header("Bienvenue sur la page d'accueil !")
    
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Parlez maintenant...")
        audio = r.listen(source)
        
    try:
        text = r.recognize_google(audio, language='en-US')# change in french #'fr-FR'
        st.write("Vous avez dit: " + text)
    except sr.UnknownValueError:
        st.write("Google Speech Recognition n'a pas pu comprendre l'audio")
    except sr.RequestError as e:
        st.write("Impossible d'obtenir les résultats de Google Speech Recognition ; {0}".format(e))


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
# Créer une liste des pages
pages = {
    "Accueil 🏠": home,
    "Page 1 📺": page1,
    "Page 2 🛡": page2,
    "Page 3 ⚔": page3
}

# Ajouter une barre de navigation pour naviguer entre les pages
st.sidebar.title("Menu 📊")
selection = st.sidebar.radio("Aller à", list(pages.keys()))

# Exécuter la fonction de la page sélectionnée
page = pages[selection]
page()