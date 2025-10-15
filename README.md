# Calibration-Fisheye-DeepLearning
Rectification d’images fisheye par Deep Learning (U-Net et cGAN)

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

---

## Structure du dépôt

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
