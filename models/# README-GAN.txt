# FisheyeGAN - Rectification et Calibration Automatique d'Images Fisheye

Un Modéle de deep learning combinant GAN et CNN pour la calibration automatique d'images fisheye et l'estimation des paramètres de calibration de caméra.

## Table des matières

- [Description courte](#description-courte)
- [Prérequis](#prérequis)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Entraînement](#entraînement)
- [Évaluation](#évaluation)
- [Utilisation](#utilisation)
- [Résultats](#résultats)
- [Notes techniques](#notes-techniques)
- [Citation](#citation)

## Description courte

Ce projet implémente une approche hybride utilisant :
- **GAN ** : Pour la rectification automatique d'images fisheye vers des images calibrées
- **CNN de calibration** : Pour l'estimation automatique des paramètres intrinsèques de caméra (focales, centre optique, distorsion)

L'architecture permet de traiter des images fisheye générales sans connaissance préalable des paramètres de distorsion, produisant simultanément une image rectifiée et les coefficients de calibration.

## Prérequis
# Environnement Python
Python >= 3.8
CUDA >= 11.0  # recommandé pour entraînement rapide sur GPU

# Dépendances principales
torch >= 1.10.0         # framework deep learning
torchvision >= 0.11.0   # transformations et modèles pré-entraînés
opencv-python >= 4.5.0  # traitement d'images
scikit-image >= 0.18.0  # métriques PSNR/SSIM, utilitaires image
matplotlib >= 3.5.0     # visualisation
pandas >= 1.3.0         # gestion CSV (labels)
numpy >= 1.21.0         # opérations matricielles
Pillow >= 8.3.0         # chargement images
tqdm >= 4.62.0          # barres de progression



# Installer toutes les dépendances en une seule commande
pip install torch torchvision opencv-python scikit-image matplotlib pandas numpy Pillow tqdm


## Architecture

### 1. GAN U-Net Generator
- **Architecture** : Encoder-Decoder avec connexions de saut (skip connections)
- **Entrée** : Image fisheye (256×256×3)
- **Sortie** : Image rectifiée (256×256×3)
- **Fonction de perte** : L1 + Adversariale

### 2. Discriminateur
- **Type** : CNN conditionnel (Patch-GAN)
- **Entrée** : Concaténation image source + image cible
- **Objectif** : Distinguer images réelles/générées

### 3. CNN de Calibration
- **Base** : ResNet-18 modifié
- **Sortie** : 8 paramètres [fx, fy, cx, cy, k1, k2, k3, k4]
- **Fonction de perte** : MSE

```
Input Fisheye (256x256x3)
         ↓
    ┌─────────────┐    ┌──────────────┐
    │ GAN U-Net   │    │ Calibration  │
    │ Generator   │    │ CNN          │
    └─────────────┘    └──────────────┘
         ↓                    ↓
  Rectified Image      Camera Parameters
   (256x256x3)         [fx,fy,cx,cy,k1-k4]
```

## Dataset

### Structure attendue
```
dataset-synth/
├── fisheye_data/
│   ├── Fisheye/
│   │   └── img/           # Images fisheye d'entrée
│   └── Calibree/
│       └── img/           # Images calibrées de référence
└── labels.csv             # Paramètres de calibration
```

### Format labels.csv
```csv
filename,fx,fy,cx,cy,k1,k2,k3,k4
image001.jpg,350.5,351.2,128.0,128.0,-0.15,0.08,-0.02,0.01
```

### Génération automatique
Si `labels.csv` n'existe pas, le script génère automatiquement des paramètres aléatoires :
- **Focales** : fx ∈ [300, 400], fy = fx ± 5
- **Centre optique** : cx = cy = 128 (centre de l'image)
- **Distorsion** : k1-k4 ∈ [-0.3, 0.3]

## Entraînement

# Hyperparamètres principaux
BATCH_SIZE = 16        # Nombre d'images par batch
NUM_EPOCHS = 120       # Nombre total d'époques
LR_GENERATOR = 2e-4    # Learning rate du générateur (plus rapide)
LR_DISCRIMINATOR = 1e-4  # Learning rate du discriminateur (plus lent)
LR_CALIBRATION = 1e-4    # Learning rate du CNN de calibration

# Pondération des pertes
λ_adversarial = 0.5    # Poids de la perte adversariale
λ_reconstruction = 50  # Reconstruction plus importante que l'adversarial

# Installer toutes les dépendances en une seule commande
pip install torch torchvision opencv-python scikit-image matplotlib pandas numpy Pillow tqdm

### Optimisations implémentées
- **Taux d'apprentissage différenciés** : G/D équilibré
- **Ratio d'entraînement G:D = 2:1** : Stabilité améliorée  
- **Label smoothing** : Évite la saturation
- **Gradient clipping** : Évite l'explosion des gradients
- **Learning rate decay** : Après 80 époques

### Lancement
# Copier le dataset vers l'espace de travail
cp -r /chemin/dataset-synth /kaggle/working/

# Lancer l'entraînement principal
python train_fisheye_gan.py

## Évaluation
import torch
from models import UNetGenerator, CalibrationCNN

# Initialiser les modèles
generator = UNetGenerator()
calibration_model = CalibrationCNN()

# Charger les poids entraînés
generator.load_state_dict(torch.load('gan_generator.pth'))
calibration_model.load_state_dict(torch.load('calibration_cnn.pth'))

# Mettre les modèles en mode évaluation
generator.eval()
calibration_model.eval()

### Métriques de qualité d'image
- **PSNR** (Peak Signal-to-Noise Ratio) : Mesure du bruit
- **SSIM** (Structural Similarity) : Similarité structurelle
- **MAE** (Mean Absolute Error) : Erreur pixellaire moyenne

### Métriques de calibration
- **MAE paramètres** : Erreur moyenne sur les 8 coefficients


## Utilisation

### Chargement des modèles
```python
import torch
from models import UNetGenerator, CalibrationCNN

# Charger les modèles entraînés
generator = UNetGenerator()
calibration_model = CalibrationCNN()

generator.load_state_dict(torch.load('gan_generator.pth'))
calibration_model.load_state_dict(torch.load('calibration_cnn.pth'))

generator.eval()
calibration_model.eval()
```

### Inférence sur image unique
from torchvision import transforms
from PIL import Image

# Préprocessing : resize + normalisation [-1,1]
transform = transforms.Compose([
    transforms.Resize((256, 256)),          # redimensionner
    transforms.ToTensor(),                  # convertir en tenseur
    transforms.Normalize([0.5]*3, [0.5]*3)  # normalisation
])

# Charger image fisheye
fisheye_img = Image.open('fisheye_input.jpg')
x = transform(fisheye_img).unsqueeze(0)  # ajouter dimension batch

with torch.no_grad():
    # Rectification image
    rectified = generator(x)
    
    # Estimation paramètres caméra
    params = calibration_model(x)
    fx, fy, cx, cy, k1, k2, k3, k4 = params[0].cpu().numpy()

print(f"Paramètres estimés: fx={fx:.2f}, fy={fy:.2f}")

### Traitement par lots

# Fonction pour traiter une séquence d'images ou vidéo
def process_fisheye_sequence(image_paths):
    results = []
    for path in image_paths:
        img = Image.open(path)
        x = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            rectified = generator(x)          # image rectifiée
            params = calibration_model(x)     # paramètres estimés
            
        results.append({
            'rectified': rectified,
            'parameters': params.cpu().numpy()
        })
    
    return results

## Résultats

### Performance finale (120 époques)
# Métriques obtenues après entraînement complet
📊 Qualité d'image:
   • PSNR: 27 dB ✅       
   • SSIM: 0.852 ✅         
   • MAE: 0.041 ✅          

📐 Calibration:
   • MAE paramètres: 0.08 ✅
   • Précision focale: ±2.1 pixels
   • Précision distorsion: ±0.03

### Comparaison visuelle
- **Temps d'inférence** : ~15ms par image (GPU)
- **Qualité rectification** :performante par rapport aux méthodes classique
- **Avantage** : Pas de calibration manuelle requise

## Notes techniques

### Optimisations d'entraînement
1. **Stabilisation GAN** : Ratio G:D adaptatif, label smoothing
2. **Convergence** : Learning rate decay progressif après 80 époques
3. **Mémoire** : Batch size optimisé pour GPU 16GB
4. **Reproductibilité** : Seeds fixes, split train/test déterministe

### Limitations
- **Résolution fixe** : Entraîné sur 256×256 (adaptable)
- **Domaine** : Optimisé pour fisheye modérée (k1 ∈ [-0.3, 0.3])
- **Généralisation** : Performance dépend de la diversité du dataset

### Extensions possibles
- **Multi-résolution** : Pyramides d'images
- **Temps réel** : Optimisation mobile/embarqué  
- **3D** : Extension aux caméras stéréo fisheye

### Fichiers générés
```
/kaggle/working/
├── gan_generator.pth           # Poids du générateur
├── gan_discriminator.pth       # Poids du discriminateur  
├── calibration_cnn.pth         # Poids du réseau de calibration
├── training_metrics_optimized.pkl  # Historique d'entraînement
└── labels.csv                  # Paramètres de calibration générés
