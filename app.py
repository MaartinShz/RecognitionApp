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
    
    
    def recognize_speech():
          with sr.Microphone() as source:
              st.write("Parlez...")
              r.adjust_for_ambient_noise(source, duration=0.5)
              audio = r.listen(source)
              try:
                  text = r.recognize_google(audio, language='fr-FR')
                  #return text
              except sr.UnknownValueError:
                  text = "Je n'ai pas compris ce que vous avez dit"
                  #st.write("Je n'ai pas compris ce que vous avez dit")
              except sr.RequestError as e:
                  text="Une erreur s'est produite : {}".format(e)
                  #st.write("Une erreur s'est produite : {}".format(e))
              return text
              
     
    while True:
          text = recognize_speech()
          st.write("Vous avez dit : {}".format(text))

    
    
    
    
    
    
    
    
    
    # Initialiser l'objet Recognizer
    # r = sr.Recognizer()
    # # Fonction de rappel pour la méthode listen_in_background()
    # def callback(recognizer, audio):
    #     try:
    #         st.write('Parlez svp ')
    #         text = recognizer.recognize_google(audio, language='fr-FR')# change in french #'fr-FR'
    #         st.write("Vous avez dit: " + text)
    #     except sr.UnknownValueError:
    #         st.write("Google Speech Recognition n'a pas pu comprendre l'audio")
    #     except sr.RequestError as e:
    #         st.write("Impossible d'obtenir les résultats de Google Speech Recognition ; {0}".format(e))
    
    # # Utiliser la méthode listen_in_background() pour écouter en continu
    # with sr.Microphone() as source:
    #     r.adjust_for_ambient_noise(source, duration=0.5) # Régler le niveau de bruit ambiant
    #     stop_listening = r.listen_in_background(source, callback)
    
    # # Attendre jusqu'à ce que l'utilisateur clique sur le bouton d'arrêt
    # stop_button = st.button("Arrêter l'écoute")
    # if stop_button:
    #     stop_listening(wait_for_stop=False)


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