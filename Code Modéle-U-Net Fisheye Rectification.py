!pip install segmentation-models-pytorch

## 3. Importations
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
import albumentations as A
import segmentation_models_pytorch as smp

## 2. Installation des dépendances
!pip install albumentations==1.3.1
!pip install segmentation-models-pytorch
!pip install --upgrade git+https://github.com/albumentations-team/albumentations.git
!pip install segmentation-models-pytorch

#### U-Net Fisheye Rectification - VERSION CORRIGÉE Optimisé ANISSA avec Sauvegarde

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
import albumentations as A
import segmentation_models_pytorch as smp
from tqdm import tqdm
import time
import random
from sklearn.metrics import mean_squared_error
import warnings
from datetime import datetime
import json
warnings.filterwarnings('ignore')

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

class FixedFisheyeDataset(Dataset):
    def __init__(self, fisheye_dir, rectified_dir, target_size=(640, 640), mode='train'):
        self.fisheye_dir = fisheye_dir
        self.rectified_dir = rectified_dir
        self.target_size = target_size
        self.mode = mode
        
        if not os.path.exists(fisheye_dir) or not os.path.exists(rectified_dir):
            raise FileNotFoundError(f"Dossiers manquants: {fisheye_dir} ou {rectified_dir}")
        
        fisheye_images = set(os.listdir(fisheye_dir))
        rectified_images = set(os.listdir(rectified_dir))
        common_images = fisheye_images.intersection(rectified_images)
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        self.images = [img for img in common_images
                      if any(img.lower().endswith(ext) for ext in valid_extensions)]
        self.images.sort()
        
        print(f"Dataset {mode} avec {len(self.images)} images")
        
        # CORRECTION PRINCIPALE: Désactiver la vérification des formes et redimensionner séparément
        if mode == 'train':
            # Transformations géométriques pour fisheye
            self.fisheye_transform = A.Compose([
                A.Resize(target_size[0], target_size[1], interpolation=cv2.INTER_LINEAR),
                A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.3),
                A.HueSaturationValue(hue_shift_limit=3, sat_shift_limit=5, val_shift_limit=5, p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
            
            # Transformations géométriques pour rectified (mêmes transformations mais appliquées séparément)
            self.rectified_transform = A.Compose([
                A.Resize(target_size[0], target_size[1], interpolation=cv2.INTER_LINEAR),
                A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.3),
                A.HueSaturationValue(hue_shift_limit=3, sat_shift_limit=5, val_shift_limit=5, p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            # Mode validation: transformations simples
            self.fisheye_transform = A.Compose([
                A.Resize(target_size[0], target_size[1], interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
            
            self.rectified_transform = A.Compose([
                A.Resize(target_size[0], target_size[1], interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        fisheye_path = os.path.join(self.fisheye_dir, img_name)
        rectified_path = os.path.join(self.rectified_dir, img_name)
        
        fisheye_img = cv2.imread(fisheye_path)
        rectified_img = cv2.imread(rectified_path)
        
        if fisheye_img is None or rectified_img is None:
            raise ValueError(f"Impossible de charger: {img_name}")
        
        # Convertir BGR -> RGB
        fisheye_img = cv2.cvtColor(fisheye_img, cv2.COLOR_BGR2RGB)
        rectified_img = cv2.cvtColor(rectified_img, cv2.COLOR_BGR2RGB)
        
        # CORRECTION: Appliquer les transformations séparément avec le même seed
        if self.mode == 'train':
            # Utiliser le même seed pour les deux images pour synchroniser les transformations aléatoires
            seed = random.randint(0, 2**32 - 1)
            
            # Appliquer les transformations à fisheye
            random.seed(seed)
            np.random.seed(seed)
            fisheye_transformed = self.fisheye_transform(image=fisheye_img)['image']
            
            # Appliquer les mêmes transformations à rectified
            random.seed(seed)
            np.random.seed(seed)
            rectified_transformed = self.rectified_transform(image=rectified_img)['image']
        else:
            # Mode validation: pas de transformations aléatoires
            fisheye_transformed = self.fisheye_transform(image=fisheye_img)['image']
            rectified_transformed = self.rectified_transform(image=rectified_img)['image']
        
        return fisheye_transformed, rectified_transformed

def create_data_loaders(fisheye_dir, rectified_dir, batch_size=4, val_ratio=0.2):
    """DataLoaders avec split simple"""
    
    # Dataset pour obtenir la liste des images
    full_dataset = FixedFisheyeDataset(fisheye_dir, rectified_dir, mode='val')
    
    # Split simple
    total_size = len(full_dataset.images)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    
    # Mélanger et diviser
    indices = list(range(total_size))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Créer les datasets
    train_dataset = FixedFisheyeDataset(fisheye_dir, rectified_dir, mode='train')
    val_dataset = FixedFisheyeDataset(fisheye_dir, rectified_dir, mode='val')
    
    # Appliquer le split
    train_dataset.images = [full_dataset.images[i] for i in train_indices]
    val_dataset.images = [full_dataset.images[i] for i in val_indices]
    
    print(f"Split: {len(train_dataset)} train, {len(val_dataset)} validation")
    
    # DataLoaders avec paramètres optimisés
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # Réduit à 0 pour éviter les problèmes de multiprocessing
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,  # Réduit à 0 pour éviter les problèmes de multiprocessing
        pin_memory=True
    )
    
    return train_loader, val_loader

class PerceptualLoss(nn.Module):
    """Loss perceptuelle pour de meilleurs résultats visuels"""
    def __init__(self, mse_weight=0.8, l1_weight=0.2):
        super(PerceptualLoss, self).__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)
        l1 = self.l1_loss(pred, target)
        return self.mse_weight * mse + self.l1_weight * l1

def create_model(device):
    """Modèle avec activation finale appropriée"""
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=3,
        activation=None  # Pas d'activation finale - on gère manuellement
    )
    
    # Modifier la sortie finale pour s'assurer qu'elle est dans la bonne plage
    class UNetWithTanh(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            
        def forward(self, x):
            # Obtenir la sortie du modèle de base
            output = self.base_model(x)
            # Appliquer tanh pour avoir une sortie dans [-1, 1]
            # Compatible avec la normalisation ImageNet
            return torch.tanh(output)
    
    model = UNetWithTanh(model)
    model = model.to(device)
    
    print(f"Modèle créé sur {device}")
    print(f"Nombre de paramètres: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

def calculate_metrics(pred, target):
    """Calcul des métriques de qualité"""
    # Dénormaliser pour le calcul du PSNR
    pred_denorm = pred * 0.5 + 0.5  # Approximation de dénormalisation
    target_denorm = target * 0.5 + 0.5
    
    # Clipper entre 0 et 1
    pred_denorm = torch.clamp(pred_denorm, 0, 1)
    target_denorm = torch.clamp(target_denorm, 0, 1)
    
    # MSE
    mse = torch.mean((pred_denorm - target_denorm) ** 2).item()
    
    # PSNR
    if mse > 0:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    else:
        psnr = float('inf')
    
    # SSIM approximé (corrélation structurelle)
    def ssim_approx(x, y):
        mu_x = torch.mean(x)
        mu_y = torch.mean(y)
        sigma_x = torch.std(x)
        sigma_y = torch.std(y)
        sigma_xy = torch.mean((x - mu_x) * (y - mu_y))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2))
        return ssim.item()
    
    ssim = ssim_approx(pred_denorm, target_denorm)
    
    # MAE
    mae = torch.mean(torch.abs(pred_denorm - target_denorm)).item()
    
    return {'mse': mse, 'psnr': psnr, 'ssim': ssim, 'mae': mae}

def create_results_directory():
    """Créer un dossier de résultats avec timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"fisheye_results_{timestamp}"
    
    # Créer les sous-dossiers
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "models"), exist_ok=True)
    
    print(f"📁 Dossier de résultats créé: {results_dir}")
    return results_dir

def train_model(model, train_loader, val_loader, device, results_dir, max_epochs=100):
    """Entraînement avec paramètres ajustés et sauvegarde des métriques"""
    
    # Configuration d'entraînement optimisée
    criterion = PerceptualLoss(mse_weight=0.8, l1_weight=0.2)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)
    
    # Listes pour stocker les métriques
    train_losses = []
    val_losses = []
    val_psnrs = []
    val_ssims = []
    val_maes = []
    learning_rates = []
    best_val_loss = float('inf')
    
    print(f"\n🚀 Entraînement (max {max_epochs} epochs)")
    print(f"Loss: MSE(0.8) + L1(0.2), Patience: 15, LR: 1e-4")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print("-" * 80)
    
    for epoch in range(max_epochs):
        # === ENTRAÎNEMENT ===
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1:2d}/{max_epochs} [Train]')
        
        for fisheye, rectified in train_pbar:
            fisheye, rectified = fisheye.to(device), rectified.to(device)
            
            optimizer.zero_grad()
            output = model(fisheye)
            loss = criterion(output, rectified)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.5f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # === VALIDATION ===
        model.eval()
        val_loss = 0.0
        val_batches = 0
        total_psnr = 0.0
        total_ssim = 0.0
        total_mae = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1:2d}/{max_epochs} [Val]   ')
            
            for fisheye, rectified in val_pbar:
                fisheye, rectified = fisheye.to(device), rectified.to(device)
                
                output = model(fisheye)
                loss = criterion(output, rectified)
                
                metrics = calculate_metrics(output, rectified)
                
                val_loss += loss.item()
                total_psnr += metrics['psnr'] if metrics['psnr'] != float('inf') else 0
                total_ssim += metrics['ssim']
                total_mae += metrics['mae']
                val_batches += 1
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.5f}',
                    'PSNR': f'{metrics["psnr"]:.1f}' if metrics['psnr'] != float('inf') else '∞',
                    'SSIM': f'{metrics["ssim"]:.3f}'
                })
        
        avg_val_loss = val_loss / val_batches
        avg_psnr = total_psnr / val_batches
        avg_ssim = total_ssim / val_batches
        avg_mae = total_mae / val_batches
        
        val_losses.append(avg_val_loss)
        val_psnrs.append(avg_psnr)
        val_ssims.append(avg_ssim)
        val_maes.append(avg_mae)
        
        # Scheduler step
        scheduler.step()
        
        # Sauvegarde du meilleur modèle
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'psnr': avg_psnr,
                'ssim': avg_ssim,
                'mae': avg_mae,
            }, os.path.join(results_dir, 'models', 'best_fisheye_model.pth'))
        
        # Monitoring
        overfitting_ratio = avg_val_loss / avg_train_loss
        print(f"Epoch {epoch+1:2d} | "
              f"Train: {avg_train_loss:.6f} | "
              f"Val: {avg_val_loss:.6f} | "
              f"PSNR: {avg_psnr:5.2f} | "
              f"SSIM: {avg_ssim:.3f} | "
              f"MAE: {avg_mae:.4f} | "
              f"Ratio: {overfitting_ratio:.3f}")
        
        # Early stopping
        if early_stopping(avg_val_loss, model):
            print(f"\n🛑 Early stopping à l'epoch {epoch+1}")
            break
        
        print("-" * 80)
    
    # Sauvegarder les métriques
    metrics_data = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_psnrs': val_psnrs,
        'val_ssims': val_ssims,
        'val_maes': val_maes,
        'learning_rates': learning_rates,
        'epochs': list(range(1, len(train_losses) + 1))
    }
    
    with open(os.path.join(results_dir, 'training_metrics.json'), 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    return train_losses, val_losses, val_psnrs, val_ssims, val_maes, learning_rates

def denormalize_tensor(tensor):
    """Dénormaliser un tensor ImageNet"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    if tensor.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    
    denorm = tensor * std + mean
    return torch.clamp(denorm, 0, 1)

def save_sample_results(model, val_loader, device, results_dir, num_samples=6):
    """Sauvegarder les résultats visuels avec métriques détaillées"""
    model.eval()
    
    fisheye_batch, rectified_batch = next(iter(val_loader))
    fisheye_batch = fisheye_batch.to(device)
    rectified_batch = rectified_batch.to(device)
    
    with torch.no_grad():
        predictions = model(fisheye_batch)
    
    # Dénormaliser pour l'affichage
    fisheye_denorm = denormalize_tensor(fisheye_batch.cpu())
    rectified_denorm = denormalize_tensor(rectified_batch.cpu())
    predictions_denorm = denormalize_tensor(predictions.cpu())
    
    # Créer une grande figure avec toutes les comparaisons
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 6 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(min(num_samples, fisheye_batch.size(0))):
        # Calculer les métriques pour cette image
        pred_img = predictions_denorm[i]
        gt_img = rectified_denorm[i]
        
        # Recalculer les métriques avec les tenseurs dénormalisés
        mse = torch.mean((pred_img - gt_img) ** 2).item()
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        mae = torch.mean(torch.abs(pred_img - gt_img)).item()
        
        # SSIM approximé
        mu_pred = torch.mean(pred_img)
        mu_gt = torch.mean(gt_img)
        sigma_pred = torch.std(pred_img)
        sigma_gt = torch.std(gt_img)
        sigma_cross = torch.mean((pred_img - mu_pred) * (gt_img - mu_gt))
        
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim = ((2 * mu_pred * mu_gt + c1) * (2 * sigma_cross + c2)) / \
               ((mu_pred ** 2 + mu_gt ** 2 + c1) * (sigma_pred ** 2 + sigma_gt ** 2 + c2))
        ssim = ssim.item()
        
        # Affichage
        axes[i, 0].imshow(fisheye_denorm[i].permute(1, 2, 0))
        axes[i, 0].set_title(f'Image Fisheye Originale #{i+1}', fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(rectified_denorm[i].permute(1, 2, 0))
        axes[i, 1].set_title(f'Ground Truth #{i+1}', fontsize=12, fontweight='bold')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(predictions_denorm[i].permute(1, 2, 0))
        axes[i, 2].set_title(f'Image Rectifiée (Prédiction) #{i+1}\nPSNR: {psnr:.1f}dB | SSIM: {ssim:.3f}\nMSE: {mse:.4f} | MAE: {mae:.4f}', 
                           fontsize=10, fontweight='bold')
        axes[i, 2].axis('off')
        
        # Sauvegarder les images individuelles
        plt.imsave(os.path.join(results_dir, 'images', f'fisheye_{i+1}.png'), 
                   fisheye_denorm[i].permute(1, 2, 0).numpy())
        plt.imsave(os.path.join(results_dir, 'images', f'ground_truth_{i+1}.png'), 
                   rectified_denorm[i].permute(1, 2, 0).numpy())
        plt.imsave(os.path.join(results_dir, 'images', f'prediction_{i+1}.png'), 
                   predictions_denorm[i].permute(1, 2, 0).numpy())
    
    plt.suptitle('Comparaison des Résultats de Rectification Fisheye', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'images', 'comparison_results.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Images sauvegardées dans {os.path.join(results_dir, 'images')}")

def save_training_plots(train_losses, val_losses, val_psnrs, val_ssims, val_maes, learning_rates, results_dir):
    """Sauvegarder tous les graphiques d'entraînement"""
    
    # Figure principale avec 6 sous-graphiques
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Métriques d\'Entraînement - U-Net Fisheye Rectification', fontsize=16, fontweight='bold')
    
    epochs = list(range(1, len(train_losses) + 1))
    
    # 1. Losses
    axes[0, 0].plot(epochs, train_losses, label='Train Loss', color='blue', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, label='Validation Loss', color='red', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Évolution des Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. PSNR
    axes[0, 1].plot(epochs, val_psnrs, label='Validation PSNR', color='green', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('PSNR (dB)')
    axes[0, 1].set_title('Peak Signal-to-Noise Ratio')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. SSIM
    axes[0, 2].plot(epochs, val_ssims, label='Validation SSIM', color='purple', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('SSIM')
    axes[0, 2].set_title('Structural Similarity Index')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. MAE
    axes[1, 0].plot(epochs, val_maes, label='Validation MAE', color='orange', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].set_title('Mean Absolute Error')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Learning Rate
    axes[1, 1].plot(epochs, learning_rates, label='Learning Rate', color='brown', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Évolution du Learning Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    # 6. Overfitting Monitor
    if len(train_losses) > 0:
        ratios = [val/train for val, train in zip(val_losses, train_losses)]
        axes[1, 2].plot(epochs, ratios, label='Val/Train Loss Ratio', color='red', linewidth=2)
        axes[1, 2].axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Ratio = 1')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Ratio')
        axes[1, 2].set_title('Monitoring de l\'Overfitting')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plots', 'training_metrics.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Graphiques individuels pour une meilleure lisibilité
    
    # Loss individuel
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, val_losses, label='Validation Loss', color='red', linewidth=2, marker='s', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Évolution des Losses d\'Entraînement', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plots', 'losses.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # PSNR individuel
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_psnrs, label='Validation PSNR', color='green', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title('Évolution du Peak Signal-to-Noise Ratio', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plots', 'psnr.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # SSIM individuel
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_ssims, label='Validation SSIM', color='purple', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('SSIM', fontsize=12)
    plt.title('Évolution du Structural Similarity Index', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plots', 'ssim.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Graphiques sauvegardés dans {os.path.join(results_dir, 'plots')}")

def save_final_report(results_dir, train_losses, val_losses, val_psnrs, val_ssims, val_maes, model_params):
    """Créer un rapport final de l'entraînement"""
    
    final_metrics = {
        'final_train_loss': train_losses[-1] if train_losses else 0,
        'final_val_loss': val_losses[-1] if val_losses else 0,
        'best_val_loss': min(val_losses) if val_losses else 0,
        'final_psnr': val_psnrs[-1] if val_psnrs else 0,
        'best_psnr': max(val_psnrs) if val_psnrs else 0,
        'final_ssim': val_ssims[-1] if val_ssims else 0,
        'best_ssim': max(val_ssims) if val_ssims else 0,
        'final_mae': val_maes[-1] if val_maes else 0,
        'best_mae': min(val_maes) if val_maes else 0,
        'total_epochs': len(train_losses),
        'model_parameters': model_params
    }
    
    # Rapport textuel
    report = f"""
# RAPPORT D'ENTRAÎNEMENT - U-NET FISHEYE RECTIFICATION
===============================================================

## Résumé des Performances
- **Epochs d'entraînement**: {final_metrics['total_epochs']}
- **Paramètres du modèle**: {final_metrics['model_parameters']:,}

## Métriques Finales
- **Loss d'entraînement final**: {final_metrics['final_train_loss']:.6f}
- **Loss de validation final**: {final_metrics['final_val_loss']:.6f}
- **Meilleure loss de validation**: {final_metrics['best_val_loss']:.6f}

## Qualité d'Image
- **PSNR final**: {final_metrics['final_psnr']:.2f} dB
- **Meilleur PSNR**: {final_metrics['best_psnr']:.2f} dB
- **SSIM final**: {final_metrics['final_ssim']:.4f}
- **Meilleur SSIM**: {final_metrics['best_ssim']:.4f}
- **MAE final**: {final_metrics['final_mae']:.6f}
- **Meilleur MAE**: {final_metrics['best_mae']:.6f}


## Fichiers Générés
- `models/best_fisheye_model.pth`: Meilleur modèle sauvegardé
- `images/`: Images de comparaison et résultats visuels
- `plots/`: Graphiques des métriques d'entraînement
- `training_metrics.json`: Données numériques complètes

Rapport généré le: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
===============================================================
"""
    
    # Sauvegarder le rapport
    with open(os.path.join(results_dir, 'training_report.txt'), 'w') as f:
        f.write(report)
    
    # Sauvegarder les métriques finales en JSON
    with open(os.path.join(results_dir, 'final_metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    print("📊 Rapport final généré")
    print(report)
    
    return final_metrics

def debug_first_batch(train_loader, val_loader):
    """Debug détaillé du premier batch"""
    print("\n🔍 Debug détaillé du dataset...")
    
    try:
        for name, loader in [("Train", train_loader), ("Val", val_loader)]:
            fisheye, rectified = next(iter(loader))
            
            print(f"\n{name} Dataset:")
            print(f"  Fisheye shape: {fisheye.shape}")
            print(f"  Rectified shape: {rectified.shape}")
            print(f"  Fisheye range: [{fisheye.min():.3f}, {fisheye.max():.3f}]")
            print(f"  Rectified range: [{rectified.min():.3f}, {rectified.max():.3f}]")
            
            # Différence moyenne
            diff = torch.mean(torch.abs(fisheye - rectified)).item()
            print(f"  Différence moyenne: {diff:.3f}")
            
            if diff < 0.1:
                print(f"  ⚠️  ATTENTION: Différence très faible entre fisheye et rectified!")
                
        print("✅ Debug terminé avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur lors du debug: {e}")
        raise

# SCRIPT PRINCIPAL
if __name__ == "__main__":
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Créer le dossier de résultats
    results_dir = create_results_directory()
    
    # Chemins
    fisheye_dir = '/kaggle/input/dataset-synthtique/fisheye_data/Fisheye/img'
    rectified_dir = '/kaggle/input/dataset-synthtique/fisheye_data/Calibree/img'
    
    # Créer les DataLoaders
    print("\n📁 Création des DataLoaders...")
    try:
        train_loader, val_loader = create_data_loaders(
            fisheye_dir, 
            rectified_dir, 
            batch_size=8,  # Batch size plus petit pour stabilité
            val_ratio=0.15
        )
        
        # Debug
        debug_first_batch(train_loader, val_loader)
        
    except Exception as e:
        print(f"❌ Erreur lors de la création des DataLoaders: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Créer le modèle
    print("\n🏗️ Création du modèle...")
    model = create_model(device)
    model_params = sum(p.numel() for p in model.parameters())
    
    # Entraînement
    print("\n🎯 Lancement de l'entraînement...")
    train_losses, val_losses, val_psnrs, val_ssims, val_maes, learning_rates = train_model(
        model, 
        train_loader, 
        val_loader, 
        device, 
        results_dir,
        max_epochs=100
    )
    
    # Sauvegarder les résultats visuels
    print("\n📊 Sauvegarde des résultats visuels...")
    save_sample_results(model, val_loader, device, results_dir, num_samples=6)
    
    # Sauvegarder les graphiques d'entraînement
    print("\n📈 Génération des graphiques de métriques...")
    save_training_plots(train_losses, val_losses, val_psnrs, val_ssims, val_maes, learning_rates, results_dir)
    
    # Générer le rapport final
    print("\n📋 Génération du rapport final...")
    final_metrics = save_final_report(results_dir, train_losses, val_losses, val_psnrs, val_ssims, val_maes, model_params)
    
    print(f"\n🎉 ENTRAÎNEMENT TERMINÉ!")
    print(f"📁 Tous les résultats sont sauvegardés dans: {results_dir}")
    print(f"🏆 Meilleur PSNR: {final_metrics['best_psnr']:.2f} dB")
    print(f"🏆 Meilleur SSIM: {final_metrics['best_ssim']:.4f}")
    print(f"🏆 Meilleure Loss: {final_metrics['best_val_loss']:.6f}")