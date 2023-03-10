import os
import face_recognition
import pickle

# Dossier contenant les images
path = os.getcwd()
print(path)

image_dir = str(path) + "/etu/"
encoded_dir = str(path) + "/dossier_encoded/"
face_encoding = []

if not os.path.exists(encoded_dir):
    os.makedirs(encoded_dir)

# Liste des noms des fichiers JPEG dans le dossier image_dir
image_names = [f for f in os.listdir(image_dir) if f.endswith(".jpeg")]
print(image_names)

known_face_encodings = []
known_face_names = []

for image_file in image_names:
    print(image_file.split(".")[0])
    image = face_recognition.load_image_file(str(path)+"/etu/"+image_file)
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(image_file.split(".")[0])

# Enregistrer les listes dans un fichier pickle
with open(str(encoded_dir)+'known_faces.pkl', 'wb') as f:
    pickle.dump((known_face_encodings, known_face_names), f)


