# ✅ Copier tout le dossier depuis Kaggle Input vers le dossier de travail
!cp -r /kaggle/input/dataset-synth /kaggle/working/
print("✅ Dossier 'dataset-synth' copié dans /kaggle/working/")
!ls /kaggle/working/dataset-synth

# ✅ Cellule 1 : Imports & setup
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Device:", device)
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms

# Chemins
fisheye_dir = "/kaggle/working/dataset-synth/fisheye_data/Fisheye/img"
cam1_dir = "/kaggle/working/dataset-synth/fisheye_data/Calibree/img"
label_path = "/kaggle/working/labels.csv"

# Générer labels.csv si nécessaire
if not os.path.exists(label_path):
    # Filtrer uniquement les fichiers image (ignorer le dossier "depth" ou autres)
    fisheye_files = sorted([
        f for f in os.listdir(fisheye_dir)
        if os.path.isfile(os.path.join(fisheye_dir, f)) and f.endswith((".jpg", ".png"))
    ])
    
    cam1_files = set([
        f for f in os.listdir(cam1_dir)
        if os.path.isfile(os.path.join(cam1_dir, f)) and f.endswith((".jpg", ".png"))
    ])

    rows = []
    for f in fisheye_files:
        if f in cam1_files:
            fx = np.random.uniform(300, 400)
            fy = fx + np.random.uniform(-5, 5)
            cx = cy = 128
            dist = np.random.uniform(-0.3, 0.3, size=4)
            rows.append([f, fx, fy, cx, cy] + dist.tolist())

    cols = ["filename", "fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(label_path, index=False)
    print(f"✅ labels.csv généré avec {len(df)} lignes")
else:
    df = pd.read_csv(label_path)
    print(f"✅ labels.csv chargé avec {len(df)} lignes")

# Split 90% train / 10% test
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# Dataset personnalisé
class FullFisheyeDataset(Dataset):
    def __init__(self, df, fisheye_dir, cam1_dir):
        self.df = df.reset_index(drop=True)
        self.fisheye_dir = fisheye_dir
        self.cam1_dir = cam1_dir
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row["filename"]
        fx, fy, cx, cy, k1, k2, k3, k4 = row[1:].values.astype(np.float32)

        x_path = os.path.join(self.fisheye_dir, fname)
        y_path = os.path.join(self.cam1_dir, fname)

        x = Image.open(x_path).convert("RGB")
        y = Image.open(y_path).convert("RGB")

        return self.transform(x), self.transform(y), torch.tensor([fx, fy, cx, cy, k1, k2, k3, k4])

# Loaders
train_ds = FullFisheyeDataset(train_df, fisheye_dir, cam1_dir)
test_ds = FullFisheyeDataset(test_df, fisheye_dir, cam1_dir)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=1)

print(f"📦 Train: {len(train_ds)} | Test: {len(test_ds)}")
loader = DataLoader(test_ds, batch_size=1)

print(f"📦 Train: {len(train_ds)} | Test: {len(test_ds)}")

# ✅ Cellule 3 : Générateur U-Net + Réseau de calibration
class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        def down_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, 2, 1), nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2)
            )
        def up_block(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1), nn.BatchNorm2d(out_c), nn.ReLU()
            )

        self.down1 = down_block(3, 64)
        self.down2 = down_block(64, 128)
        self.down3 = down_block(128, 256)
        self.down4 = down_block(256, 512)

        self.middle = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.ReLU())

        self.up1 = up_block(512, 512)
        self.up2 = up_block(512+512, 256)
        self.up3 = up_block(256+256, 128)
        self.up4 = up_block(128+128, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(64+64, 3, 4, 2, 1), nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        m = self.middle(d4)
        u1 = self.up1(m)
        u2 = self.up2(torch.cat([u1, d4], dim=1))
        u3 = self.up3(torch.cat([u2, d3], dim=1))
        u4 = self.up4(torch.cat([u3, d2], dim=1))
        return self.final(torch.cat([u4, d1], dim=1))

# Calibration CNN (basé ResNet)
class CalibrationCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet18(weights=None)
        self.base.fc = nn.Linear(self.base.fc.in_features, 8)  # fx, fy, cx, cy, k1-k4

    def forward(self, x):
        return self.base(x)

print("✅ Modèles définis")

# ✅ Cellule 4 : Discriminateur + pertes
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4, 1, 1)
        )
    def forward(self, a, b):
        return self.net(torch.cat([a, b], dim=1))

G = UNetGenerator().to(device)
D = Discriminator().to(device)
model_calib = CalibrationCNN().to(device)

# ✅ OPTIMISEURS MODIFIÉS - TAUX DIFFÉRENCIÉS
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))  # Plus rapide
opt_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))  # Plus lent
opt_C = torch.optim.Adam(model_calib.parameters(), lr=1e-4)

bce = nn.BCEWithLogitsLoss()
l1 = nn.L1Loss()
mse = nn.MSELoss()

# ✅ FONCTION LABEL SMOOTHING
def smooth_labels(labels, smoothing=0.1):
    """Applique un lissage des labels pour stabiliser l'entraînement"""
    if smoothing == 0:
        return labels
    return labels * (1.0 - smoothing) + 0.5 * smoothing

print("✅ Optimiseurs configurés avec taux différenciés")

# ✅ Cellule 5 : Entraînement OPTIMISÉ GAN + CalibrationCNN
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Fonction pour calculer PSNR
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# Fonction pour calculer SSIM (conversion en numpy nécessaire)
def calculate_ssim_batch(img1, img2):
    """Calcule SSIM moyen sur un batch"""
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()
    
    ssim_values = []
    for i in range(img1_np.shape[0]):
        # Conversion de (C, H, W) vers (H, W, C) si nécessaire
        if img1_np.shape[1] == 3:  # RGB
            im1 = np.transpose(img1_np[i], (1, 2, 0))
            im2 = np.transpose(img2_np[i], (1, 2, 0))
            ssim_val = ssim(im1, im2, data_range=1.0, channel_axis=-1)
        else:  # Grayscale
            im1 = img1_np[i, 0]
            im2 = img2_np[i, 0]
            ssim_val = ssim(im1, im2, data_range=1.0)
        ssim_values.append(ssim_val)
    
    return np.mean(ssim_values)

# Initialisation des historiques
history_G, history_D, history_C = [], [], []
history_psnr, history_mae, history_ssim = [], [], []

# ✅ NOMBRE D'ÉPOQUES OPTIMAL POUR KAGGLE
NUM_EPOCHS = 120  

print(f"🚀 Démarrage de l'entraînement optimisé pour {NUM_EPOCHS} époques")
print("🔧 Améliorations appliquées :")
print("   • Taux d'apprentissage différenciés (G=2e-4, D=1e-4)")
print("   • Ratio d'entraînement G:D = 2:1")
print("   • Label smoothing + Gradient clipping")
print("   • Pondération adversariale réduite")

for epoch in range(NUM_EPOCHS):
    G.train(); D.train(); model_calib.train()
    total_g, total_d, total_c = 0, 0, 0
    total_psnr, total_mae, total_ssim = 0, 0, 0
    batch_count = 0

    for batch_idx, (x, y, params) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        x, y, params = x.to(device), y.to(device), params.to(device)
        fake = G(x)

        # ==========================================
        # DISCRIMINATEUR AVEC STABILISATION
        # ==========================================
        # Entraîner le discriminateur tous les 2 batches (ratio 2:1)
        if batch_idx % 2 == 0:
            real_pred = D(x, y)
            fake_pred = D(x, fake.detach())
            
            # Labels avec smoothing pour stabilité
            real_labels = smooth_labels(torch.ones_like(real_pred), smoothing=0.1)
            fake_labels = smooth_labels(torch.zeros_like(fake_pred), smoothing=0.1)
            
            d_real_loss = bce(real_pred, real_labels)
            d_fake_loss = bce(fake_pred, fake_labels)
            d_loss = 0.5 * (d_real_loss + d_fake_loss)
            
            # Gradient clipping pour éviter explosions
            opt_D.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
            opt_D.step()
        else:
            d_loss = torch.tensor(0.0)

        # ==========================================
        # GÉNÉRATEUR AVEC PONDÉRATION AJUSTÉE
        # ==========================================
        pred_fake = D(x, fake)
        
        # Pondération réduite pour équilibrer
        adversarial_loss = bce(pred_fake, torch.ones_like(pred_fake))
        reconstruction_loss = l1(fake, y)
        
        # Coefficients ajustés pour stabilité
        g_loss = 0.5 * adversarial_loss + 50 * reconstruction_loss
        
        opt_G.zero_grad()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
        opt_G.step()

        # ==========================================
        # CALIBRATIONCNN (inchangé)
        # ==========================================
        pred_params = model_calib(x)
        c_loss = mse(pred_params, params)
        opt_C.zero_grad()
        c_loss.backward()
        opt_C.step()

        # Calcul des métriques de qualité d'image
        with torch.no_grad():
            psnr_val = calculate_psnr(fake, y)
            mae_val = torch.mean(torch.abs(fake - y))
            ssim_val = calculate_ssim_batch(fake, y)

        # Accumulation des pertes et métriques
        total_d += d_loss.item()
        total_g += g_loss.item()
        total_c += c_loss.item()
        total_psnr += psnr_val.item() if torch.isfinite(psnr_val) else 50.0
        total_mae += mae_val.item()
        total_ssim += ssim_val
        batch_count += 1

    # ==========================================
    # LEARNING RATE DECAY PROGRESSIF
    # ==========================================
    if epoch > 80:  # Après 80 époques
        for param_group in opt_D.param_groups:
            param_group['lr'] *= 0.995
        for param_group in opt_G.param_groups:
            param_group['lr'] *= 0.998

    # Moyennes sur l'epoch
    avg_g = total_g / len(train_loader)
    avg_d = total_d / len(train_loader)
    avg_c = total_c / len(train_loader)
    avg_psnr = total_psnr / batch_count
    avg_mae = total_mae / batch_count
    avg_ssim = total_ssim / batch_count

    # Stocker pour le tracé
    history_G.append(avg_g)
    history_D.append(avg_d)
    history_C.append(avg_c)
    history_psnr.append(avg_psnr)
    history_mae.append(avg_mae)
    history_ssim.append(avg_ssim)

    # Affichage avec indicateurs de progression
    if (epoch + 1) % 10 == 0 or epoch < 10:
        print(f"📉 Epoch {epoch+1}/{NUM_EPOCHS} | G_loss: {avg_g:.4f} | D_loss: {avg_d:.4f} | C_loss: {avg_c:.4f}")
        print(f"📊 PSNR: {avg_psnr:.2f}dB | MAE: {avg_mae:.4f} | SSIM: {avg_ssim:.4f}")

# 📊 Visualisation complète des métriques
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Première ligne : Pertes
axes[0,0].plot(history_G, 'b-', linewidth=2, label='Generator Loss')
axes[0,0].set_title('Perte du Générateur', fontsize=12, fontweight='bold')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('G_loss')
axes[0,0].grid(True, alpha=0.3)
axes[0,0].legend()

axes[0,1].plot(history_D, 'r-', linewidth=2, label='Discriminator Loss')
axes[0,1].set_title('Perte du Discriminateur', fontsize=12, fontweight='bold')
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('D_loss')
axes[0,1].grid(True, alpha=0.3)
axes[0,1].legend()

axes[0,2].plot(history_C, 'g-', linewidth=2, label='Calibration Loss')
axes[0,2].set_title('Perte de Calibration', fontsize=12, fontweight='bold')
axes[0,2].set_xlabel('Epoch')
axes[0,2].set_ylabel('C_loss')
axes[0,2].grid(True, alpha=0.3)
axes[0,2].legend()

# Deuxième ligne : Métriques de qualité
axes[1,0].plot(history_psnr, 'purple', linewidth=2, label='PSNR')
axes[1,0].set_title('PSNR (Peak Signal-to-Noise Ratio)', fontsize=12, fontweight='bold')
axes[1,0].set_xlabel('Epoch')
axes[1,0].set_ylabel('PSNR (dB)')
axes[1,0].grid(True, alpha=0.3)
axes[1,0].legend()

axes[1,1].plot(history_mae, 'orange', linewidth=2, label='MAE')
axes[1,1].set_title('MAE (Mean Absolute Error)', fontsize=12, fontweight='bold')
axes[1,1].set_xlabel('Epoch')
axes[1,1].set_ylabel('MAE')
axes[1,1].grid(True, alpha=0.3)
axes[1,1].legend()

axes[1,2].plot(history_ssim, 'cyan', linewidth=2, label='SSIM')
axes[1,2].set_title('SSIM (Structural Similarity)', fontsize=12, fontweight='bold')
axes[1,2].set_xlabel('Epoch')
axes[1,2].set_ylabel('SSIM')
axes[1,2].grid(True, alpha=0.3)
axes[1,2].legend()

plt.tight_layout()
plt.suptitle('Métriques d\'Entraînement GAN Optimisé + CalibrationCNN', fontsize=16, fontweight='bold', y=1.02)
plt.show()

# 📊 Graphique combiné des métriques de qualité normalisées
plt.figure(figsize=(12, 6))
# Normalisation pour comparaison visuelle
norm_psnr = np.array(history_psnr) / max(history_psnr)
norm_mae = 1 - (np.array(history_mae) / max(history_mae))  # Inversé car plus bas = mieux
norm_ssim = np.array(history_ssim)

plt.plot(norm_psnr, 'purple', linewidth=2, label='PSNR (normalisé)')
plt.plot(norm_mae, 'orange', linewidth=2, label='MAE (inversé et normalisé)')
plt.plot(norm_ssim, 'cyan', linewidth=2, label='SSIM')
plt.xlabel('Epoch')
plt.ylabel('Valeur normalisée')
plt.title('Évolution des Métriques de Qualité (Normalisées)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 💾 Sauvegarder toutes les métriques
import pickle
metrics_dict = {
    'losses': {'G': history_G, 'D': history_D, 'C': history_C},
    'quality': {'PSNR': history_psnr, 'MAE': history_mae, 'SSIM': history_ssim}
}

with open('training_metrics_optimized.pkl', 'wb') as f:
    pickle.dump(metrics_dict, f)

print("✅ Entraînement optimisé terminé !")
print(f"📈 Résultats finaux (après {NUM_EPOCHS} époques):")
print(f"   📊 PSNR final: {history_psnr[-1]:.2f} dB")
print(f"   📊 MAE final: {history_mae[-1]:.4f}")
print(f"   📊 SSIM final: {history_ssim[-1]:.4f}")

# Comparaison avec objectifs
print(f"\n🎯 Évaluation des objectifs :")
print(f"   • PSNR: {history_psnr[-1]:.1f} dB {'✅' if history_psnr[-1] > 28 else '⚠️'} (objectif: >28 dB)")
print(f"   • SSIM: {history_ssim[-1]:.3f} {'✅' if history_ssim[-1] > 0.75 else '⚠️'} (objectif: >0.75)")
print(f"   • MAE: {history_mae[-1]:.4f} {'✅' if history_mae[-1] < 0.05 else '⚠️'} (objectif: <0.05)")
# ✅ Cellule 6 : Sauvegarde
torch.save(G.state_dict(), "/kaggle/working/gan_generator.pth")
torch.save(D.state_dict(), "/kaggle/working/gan_discriminator.pth")
torch.save(model_calib.state_dict(), "/kaggle/working/calibration_cnn.pth")
print("✅ Modèles sauvegardés.")
# ✅ Cellule 7 corrigée : Évaluer les modèles entraînés (GAN + CalibrationCNN)

# Charger les modèles
G = UNetGenerator().to(device)
model_calib = CalibrationCNN().to(device)
G.load_state_dict(torch.load("/kaggle/working/gan_generator.pth"))
model_calib.load_state_dict(torch.load("/kaggle/working/calibration_cnn.pth"))
G.eval(); model_calib.eval()

psnr_list = []
ssim_list = []
param_errors = []

for x, y, true_params in test_loader:
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        fake = G(x)
        pred_params = model_calib(x).cpu().numpy()[0]
        true_params = true_params.numpy()[0]

    fake_img = ((fake.squeeze(0).permute(1,2,0).cpu().numpy() + 1) / 2).clip(0, 1)
    y_img = ((y.squeeze(0).permute(1,2,0).cpu().numpy() + 1) / 2).clip(0, 1)

    # PSNR / SSIM
    psnr_list.append(psnr(y_img, fake_img, data_range=1.0))
    ssim_list.append(ssim(y_img, fake_img, data_range=1.0, channel_axis=-1))

    # Erreur sur paramètres (MAE)
    mae = np.mean(np.abs(pred_params - true_params))
    param_errors.append(mae)

# 📊 Résumé
print(f"📊 PSNR moyen : {np.mean(psnr_list):.2f}")
print(f"📊 SSIM moyen : {np.mean(ssim_list):.4f}")
print(f"📐 MAE moyenne des coefficients : {np.mean(param_errors):.4f}")
# ✅ Cellule 8 bis corrigée : Visualisation qualitative sur le test set avec modèles chargés

# Charger les modèles sauvegardés
G = UNetGenerator().to(device)
G.load_state_dict(torch.load("/kaggle/working/gan_generator.pth"))
G.eval()

model_calib = CalibrationCNN().to(device)
model_calib.load_state_dict(torch.load("/kaggle/working/calibration_cnn.pth"))
model_calib.eval()

def denorm(tensor):
    return ((tensor.permute(1, 2, 0).cpu().numpy() + 1) / 2).clip(0, 1)

# 🔍 Affichage de 3 échantillons du test set
plt.figure(figsize=(15, 9))
for i in range(3):
    x, y, true_params = test_ds[i]
    x_input = x.unsqueeze(0).to(device)

    with torch.no_grad():
        fake = G(x_input).squeeze(0)
        pred_params = model_calib(x_input).cpu().numpy()[0]

    # 🔎 Affichage
    plt.subplot(3, 3, i*3 + 1)
    plt.imshow(denorm(x)); plt.title("📷 Fisheye"); plt.axis("off")

    plt.subplot(3, 3, i*3 + 2)
    plt.imshow(denorm(fake)); plt.title("🎨 GAN Rectifiée"); plt.axis("off")

    plt.subplot(3, 3, i*3 + 3)
    plt.imshow(denorm(y)); plt.title("✅ Calibrée réelle"); plt.axis("off")

plt.tight_layout()
plt.show()
