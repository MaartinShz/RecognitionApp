FROM python:3.9

# Définir le répertoire de travail de l'application
WORKDIR /app

# Installer les dépendances système pour face_recognition
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    libopencv-dev \
    python3-dev \
    python3-numpy \
    python3-pil \
    python3-pyfftw \
    python3-scipy

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# Installer la bibliothèque face_recognition
RUN pip install --no-cache-dir face_recognition

# Copier les fichiers nécessaires de l'application dans l'image Docker
COPY . /app

EXPOSE 8501
# Exécuter la commande pour lancer l'application
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true", "--server.enableCORS", "false"]