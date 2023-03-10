# Challenge WebMining
## Reconnaissance faciale et identification sur des vidéos WEBCAM Commande vocale – Pilotage par la parole
## Présentation du challenge 

Le challenge Web Mining vise à développer une application permettant le pilotage de la webcam d'un ordinateur pour détecter les personnes présentes sur la vidéo en temps réel, et ainsi, reconnaître les étudiants faisant partie de la promotion SISE 2022-2023 en y ajoutant les informations relatives sur leur émotion, genre et âge. L'application doit être pilotée vocalement via le microphone de l'ordinateur et doit s'appuyer sur un modèle prédictif basé sur les photos d'identité étiquetées des étudiants de la promotion.

## Organisation du repository
- *requirements. txt :* Constitue l'ensemble des paquets python nécessaire en vue de l'installation et de l'execution de notre application
- `Répertoire img` : Ensemble des images intégrée au readme
- `Répertoire streamlit` : Emsemble des fichiers pour l'application
  - *Fichier app.py :* Fichier pour lancer l'application, il contient egalement les fonctions utiles à la reconnaissance vocale 
  - *Fichier face_encoding.py :* Fichier qui permet d'encoder des images
  - *Fichier webcam.py :* Fichier qui contient les fonctions utiles à la reconnaissance faciale
  - *Répertoire assets :* Ensemble des images .png intégrées à l'application (age, emotion, men, non_binary & women)
  - *Répertoire models :* Regroupe l'ensemble des modèles pré-entraînées pour la prédiction du genre, de l'âge et de l'émotion
  - *Répertoire etu :* Contient toutes les images de la base de reconnaissance
  - *Répertoire enregistrement :* Contient la vidéo enregistré si demandé



## Lancement de l'application
Soit en créant un environment et installer tous les packages "manuellement" :

```
pip install speech_recognition
pip install streamlit
pip install tensorflow
pip install --no-cache-dir -r requirements.txt

cd streamlit
streamlit run app.py

```
Vous pouvez également utiliser le fichier environnement.yml pour créer votre environnement automatiquement 

```
conda env create -f environnement.yml
conda activate challenge

cd streamlit
streamlit run app.py
```

## Fonctionnement de l'application

Notre application propose plusieurs fonctionalités que vous pourrez retrouvez dans les différents onglets. Vous pourrez utiliser notre application pour détecter les visages et les émotions des personnes présentes sur la webcam de votre ordinateur. Vous pourrez également importer ou prendre une photo afin d'alimenter notre base d'apprentissage des visages. Pour l'utiliser, cliquez sur le bouton 'Ouvrir la webcam' ci-dessous pour ouvrir la webcam de votre ordinateur et lancer l'enregistrement la vidéo.

Une fenêtre s'ouvrira et vous pourrez voir la webcam de votre ordinateur. La détection des visages et des émotions se fera en temps réel. Vous pourrez également profiter de visualisations des émotions et des âges des personnes détectées sur la vidéo. Stoppez l'affichage et l'enregistrement de la webcam en cliquant sur le bouton 'Stop webcam'. Vous pourrez ensuite télécharger la vidéo en cliquant sur le bouton 'Télécharger la vidéo' (voir panneau latéral). Nous souhaitions implémenter la reconnaissance vocale pour lancer l'enregistrement de la vidéo, mais nous n'avons pas réussi à le faire fonctionner pour toutes les fonctionnalitées. Si vous cliquez sur le bouton 'Lancer la reconnaissance vocale', vous pourrez lancer l'enregistrement de la vidéo en disant ''Démarrer la webcam''.

![pres_appli](/img/pres_appli.png)


- Alimentation des Images

Vous pouvez ajouter des images dans le dossier /etu pour alimenter notre base. Vous pouvez prendre une photo avec votre webcam qui sera importer en local dans le dossier /etu. Il vous faudra ensuite importer la photo pour la transformer et l'ajouter à notre base.

![pres_appli](/img/Alim.png)

- Reconnaissance vocale

A l'origine nous souhaitions utiliser la reconnaissance vocale pour piloter notre application. Malheureusement nous n'avons pas réussi à faire fonctionner la reconnaissance vocale pour toutes les fonctionnalitées. Dans cet onglet vous pourrez tester les différentes fonctions vocales que nous avions mis en place. Testez par exemple '' Démarrer la détection '' ou '' Arrêter l'enregistrement ''

![pres_appli](/img/Reco_vocale.png)
