

# Calibration-Fisheye-DeepLearning

Projet de correction et calibration automatique d’images fisheye à l’aide du deep learning, combinant deux approches : **U-Net** et **GAN conditionnel (Pix2Pix + CNN)**.

---

## Table des matières

- [Description du projet](#description-du-projet)  
- [Structure du dépôt](#structure-du-dépôt)  
- [Installation](#installation)  
- [Datasets et poids pré-entraînés](#datasets-et-poids-pré-entraînés)  
- [Modèles](#modèles)  
- [Usage](#usage)  
- [Licence](#licence)  

---

## Description du projet

Ce projet vise à corriger automatiquement la distorsion fisheye et à estimer les paramètres de calibration d’une caméra fisheye via deux modèles de deep learning :  

1. **U-Net** : Rectification directe d’images fisheye vers des images calibrées.  
2. **GAN conditionnel (Pix2Pix + CNN)** : Rectification + estimation simultanée des paramètres intrinsèques (fx, fy, cx, cy, k1–k4).  

L’approche permet de travailler sur des images réelles ou synthétiques sans calibration manuelle, offrant des résultats précis et rapides.
## Datasets et poids pré-entraînés

### Datasets

Téléchargez les datasets depuis Google Drive et placez-les dans le dossier `datasets/` :

- **Dataset 1** : [Google Drive](https://drive.google.com/drive/folders/1LNd4joIgr1_t6KWVl3sFWSRAGve2ubUO?usp=drive_link)  
- **Dataset 2** : [Google Drive](https://drive.google.com/drive/folders/1LNd4joIgr1_t6KWVl3sFWSRAGve2ubUO?usp=drive_link)  


---

### Poids pré-entraînés

Les fichiers `.pth` pour le **GAN** et le **U-Net** sont disponibles ici :  
[Google Drive](https://drive.google.com/drive/folders/1XjaXxvfmSOlxrZs0Qc8_0lm5qfgkENax?usp=drive_link)  
placez-les dans le dossier `weights/`.

---
### Modèles

Voir les README spécifiques :

- README_U-Net.md
- README_GAN.md

---
### Usage

- **U-Net**: rectification d’une image fisheye.
- **GAN conditionnel**: rectification + estimation des paramètres caméra.
Remarque:Consultez les README spécifiques pour les instructions détaillées

---

## Installation

### Cloner le dépôt
```bash
git clone https://github.com/Anissaoulahcerne/Calibration-Fisheye-DeepLearning.git
cd Calibration-Fisheye-DeepLearning

---

## Structure du dépôt
```bash
Calibration-Fisheye-DeepLearning/
│
├── models/
│ ├── Modéle-U-Net.py
│ ├── Modéle-GAN.py
│ ├── README_U-Net.md
│ └── README_GAN.md
│
├── weights/
│ ├── generator.pth
│ ├── discriminator.pth
│ ├── calibration.pth
│ ├── labels.txt
│ └── data.pkl
│
├── datasets/
│ ├── dataset1/
│ ├── dataset2/
│ └── dataset3/
│
├── README.md
├── requirements.txt
└── .gitignore
`### Licence`
### Licence
**Licence**

MIT License








