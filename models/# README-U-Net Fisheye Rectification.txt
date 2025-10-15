# U-Net Fisheye Rectification - Correction Automatique d'Images Fisheye

Un modèle de deep learning basé sur l'architecture U-Net pour la correction automatique d'images fisheye vers des images rectifiées, utilisant des transformations d'apprentissage supervisé.

## Table des matières

- [Description]
- [Prérequis]
- [Architecture]
- [Dataset]
- [Installation]
- [Entraînement]
- [Utilisation]
- [Résultats]
- [Structure des fichiers]
- [Notes techniques]

## Description

Ce projet implémente une approche d'apprentissage supervisé utilisant une architecture U-Net avec encoder ResNet-34 pré-entraîné pour la rectification d'images fisheye. Le modèle apprend à transformer directement des images fisheye distordues en images rectifiées sans nécessiter de paramètres de calibration explicites.

### Fonctionnalités principales
- Rectification automatique d'images fisheye
- Architecture U-Net optimisée avec skip connections
- Entraînement avec early stopping et monitoring avancé
- Métriques de qualité complètes (PSNR, SSIM, MAE)
- Sauvegarde automatique des résultats et visualisations

## Prérequis

### Environnement
- Python >= 3.8
- CUDA >= 11.0 (recommandé pour GPU)

### Dépendances principales

torch >= 1.10.0
torchvision >= 0.11.0
segmentation-models-pytorch
albumentations >= 1.3.1
opencv-python >= 4.5.0
matplotlib >= 3.5.0
numpy >= 1.21.0
tqdm >= 4.62.0
scikit-learn
```

## Architecture

### U-Net avec ResNet-34 Encoder
- **Encoder**: ResNet-34 pré-entraîné sur ImageNet
- **Decoder**: Architecture U-Net classique avec skip connections
- **Entrée**: Image fisheye (640×640×3)
- **Sortie**: Image rectifiée (640×640×3)
- **Activation finale**: Tanh pour sortie normalisée

```
Input Fisheye (640x640x3)
         ↓
    ResNet-34 Encoder
    (pré-entraîné ImageNet)
         ↓
   U-Net Decoder avec
   Skip Connections
         ↓
    Activation Tanh
         ↓
  Rectified Image (640x640x3)
```

### Fonction de perte hybride
- **Loss perceptuelle**: Combinaison MSE (80%) + L1 (20%)
- **Optimiseur**: AdamW avec weight decay
- **Scheduler**: Cosine Annealing
- **Gradient clipping**: max_norm=1.0

## Dataset

### Structure attendue
```
dataset-synth/
└── fisheye_data/
    ├── Fisheye/
    │   └── img/           # Images fisheye d'entrée
    └── Calibree/
        └── img/           # Images rectifiées de référence
```

### Formats supportés
- Extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`
- Correspondance: Noms identiques entre dossiers fisheye et rectifié
- Split automatique: 80% entraînement, 20% validation

## Installation


# Installer les dépendances principales
pip install torch torchvision
pip install segmentation-models-pytorch
pip install albumentations==1.3.1
pip install opencv-python matplotlib numpy tqdm scikit-learn

# Installation alternative via requirements (si disponible)
pip install -r requirements.txt
```

## Entraînement

### Configuration 
# Hyperparamètres principaux
BATCH_SIZE = 8
MAX_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
TARGET_SIZE = (640, 640)

# Early stopping
PATIENCE = 15
MIN_DELTA = 0.001
```

### Lancement de l'entraînement
# Définir les chemins du dataset
fisheye_dir = '/path/to/fisheye_data/Fisheye/img'
rectified_dir = '/path/to/fisheye_data/Calibree/img'


### Monitoring en temps réel
L'entraînement affiche:
- Perte d'entraînement et de validation
- Métriques PSNR, SSIM, MAE
- Ratio de surapprentissage (Val/Train Loss)
- Learning rate actuel
- Temps par epoch

## Utilisation

### Chargement du modèle entraîné

import torch
import segmentation_models_pytorch as smp
from torchvision import transforms

# Créer le modèle
model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    in_channels=3,
    classes=3,
    activation=None
)

# Wrapper avec activation Tanh
class UNetWithTanh(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, x):
        return torch.tanh(self.base_model(x))

model = UNetWithTanh(model)

# Charger les poids
checkpoint = torch.load('best_fisheye_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Inférence sur image unique
```python
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations as A

# Preprocessing
transform = A.Compose([
    A.Resize(640, 640),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Charger et traiter l'image
image = cv2.imread('fisheye_input.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Appliquer les transformations
transformed = transform(image=image)['image'].unsqueeze(0)

# Inférence
with torch.no_grad():
    rectified = model(transformed)
    
# Dénormaliser pour affichage
def denormalize_tensor(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)

rectified_img = denormalize_tensor(rectified).squeeze(0).permute(1, 2, 0).numpy()
```

## Résultats

### Performance obtenue
- **PSNR**: 23 dB (selon la complexité des distorsions)
- **SSIM**: 0.93 (similarité structurelle élevée)
- **MAE**: 0.02 (erreur pixellaire faible)
- **Temps d'inférence**: ~50ms par image (GPU RTX 3080)

### Métriques de qualité
- **Peak Signal-to-Noise Ratio (PSNR)**: Mesure du bruit résiduel
- **Structural Similarity Index (SSIM)**: Préservation des structures
- **Mean Absolute Error (MAE)**: Erreur pixellaire moyenne

## Structure des fichiers

### Fichiers générés après entraînement
```
fisheye_results_YYYYMMDD_HHMMSS/
├── models/
│   └── best_fisheye_model.pth      # Meilleur modèle sauvegardé
├── images/
│   ├── comparison_results.png       # Comparaisons visuelles
│   ├── fisheye_*.png               # Images fisheye originales
│   ├── ground_truth_*.png          # Images de référence
│   └── prediction_*.png            # Prédictions du modèle
├── plots/
│   ├── training_metrics.png        # Vue d'ensemble des métriques
│   ├── losses.png                  # Évolution des pertes
│   ├── psnr.png                   # Évolution PSNR
│   └── ssim.png                   # Évolution SSIM
├── training_metrics.json           # Données numériques complètes
├── final_metrics.json             # Métriques finales
└── training_report.txt            # Rapport textuel complet
```

## Notes techniques

### Optimisations implémentées
- **Data augmentation synchronisée**: Mêmes transformations pour paires d'images
- **Mixed precision training**: Support automatique si disponible
- **Gradient clipping**: Stabilité d'entraînement
- **Early stopping**: Évite le surapprentissage
- **Learning rate scheduling**: Convergence optimisée

### Limitations actuelles
- **Résolution fixe**: Entraîné sur 640×640 pixels
- **Domaine spécifique**: Optimisé pour distorsions modérées
- **Mémoire GPU**: Batch size limité selon VRAM disponible

### Extensions possibles
- **Multi-résolution**: Support de différentes tailles d'image
- **Temps réel**: Optimisation pour traitement vidéo
- **Architecture attention**: Mécanismes d'attention pour améliorer les performances
- **Domain adaptation**: Adaptation à différents types de caméras fisheye

### Configuration système recommandée
- **GPU**: NVIDIA RTX 3080 ou supérieur
- **RAM**: 16 GB minimum
- **VRAM**: 8 GB minimum
- **Stockage**: SSD pour chargement rapide du dataset

### Dépannage courant
- **CUDA out of memory**: Réduire le batch_size
- **Lent sur CPU**: Utiliser GPU ou réduire la résolution
- **Convergence lente**: Vérifier la qualité du dataset et l'alignement des paires