import os
import shutil
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- 0. CONFIGURATION ---
class CFG:
    # Source paths (adjust these if your raw data is elsewhere)
    raw_img_dir = Path('./train_images')
    raw_mask_dir = Path('./train_masks')
    
    # Processed paths
    root_dir = Path("./data_patches")
    
    patch_size = 518
    target_limit = 20000  # Cap for patch scanning
    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. DATA ORGANIZATION ---
def organize_data():
    """Splits raw images into train/val folders and maps .npy masks."""
    print("📂 Organizing data and splitting into Train/Val...")
    for split in ['train', 'val']:
        for cat in ['authentic', 'forged', 'masks']:
            (CFG.root_dir / split / cat).mkdir(parents=True, exist_ok=True)

    # Process Authentic
    auth_files = list((CFG.raw_img_dir / 'authentic').glob('*'))
    a_train, a_val = train_test_split(auth_files, test_size=0.2, random_state=42)
    
    # Process Forged
    forged_files = list((CFG.raw_img_dir / 'forged').glob('*'))
    f_train, f_val = train_test_split(forged_files, test_size=0.2, random_state=42)

    def distribute_files(files, split, is_forged=False):
        for fpath in files:
            dest_cat = 'forged' if is_forged else 'authentic'
            shutil.copy(fpath, CFG.root_dir / split / dest_cat / fpath.name)
            if is_forged:
                mask_name = fpath.stem + '.npy'
                m_path = CFG.raw_mask_dir / mask_name
                if m_path.exists():
                    shutil.copy(m_path, CFG.root_dir / split / 'masks' / mask_name)

    distribute_files(a_train, 'train')
    distribute_files(a_val, 'val')
    distribute_files(f_train, 'train', is_forged=True)
    distribute_files(f_val, 'val', is_forged=True)
    print("✅ Transfer Complete.")

# --- 2. TRAINING PATCH SCANNER ---
def scan_and_visualize_train(root_dir, patch_size, target_limit):
    """Scans TRAIN folder using exact stride and threshold logic."""
    print(f"\n🔍 Scanning Train patches (Size: {patch_size})...")
    train_base = root_dir / "train"
    patch_metadata = {'authentic': [], 'forged': []}

    for is_forged in [True, False]:
        category = 'forged' if is_forged else 'authentic'
        folder = train_base / category
        files = sorted([f for f in folder.glob('*') if not f.name.startswith('.')])
        
        count = 0
        for fpath in tqdm(files, desc=f"Scanning {category}"):
            if count >= target_limit: break
            try:
                if is_forged:
                    m_path = train_base / "masks" / (fpath.stem + ".npy")
                    mask = np.load(m_path)
                    if mask.ndim == 3: mask = mask[0]
                    h, w = mask.shape
                else:
                    img_info = cv2.imread(str(fpath))
                    if img_info is None: continue
                    h, w = img_info.shape[:2]

                stride = patch_size if is_forged else patch_size * 2
                
                for y in range(0, h, stride):
                    for x in range(0, w, stride):
                        if is_forged:
                            m_patch = mask[y:y+patch_size, x:x+patch_size]
                            if np.sum(m_patch > 0) < 50: continue
                        
                        patch_metadata[category].append((fpath, x, y, patch_size, is_forged))
                        count += 1
                        if count >= target_limit: break
                    if count >= target_limit: break
            except: continue

    print(f"📊 TRAIN COUNTS: Forged: {len(patch_metadata['forged'])}, Auth: {len(patch_metadata['authentic'])}")
    return patch_metadata

# --- 3. VALIDATION PROCESSING LOGIC ---
def process_val_image(img, mask=None, target_size=518):
    """Square center crop and resize/pad logic for Validation."""
    h, w = img.shape[:2]
    min_dim = min(h, w)
    start_x, start_y = (w - min_dim) // 2, (h - min_dim) // 2
    
    img_cropped = img[start_y:start_y+min_dim, start_x:start_x+min_dim]
    mask_cropped = mask[start_y:start_y+min_dim, start_x:start_x+min_dim] if mask is not None else None

    if min_dim < target_size:
        pad = target_size - min_dim
        img_final = cv2.copyMakeBorder(img_cropped, 0, pad, 0, pad, cv2.BORDER_CONSTANT, value=0)
        mask_final = cv2.copyMakeBorder(mask_cropped, 0, pad, 0, pad, cv2.BORDER_CONSTANT, value=0) if mask is not None else None
    else:
        img_final = cv2.resize(img_cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)
        mask_final = cv2.resize(mask_cropped, (target_size, target_size), interpolation=cv2.INTER_NEAREST) if mask is not None else None

    return img_final, mask_final

# --- 4. VISUALIZATION ---
def visualize_results(train_meta, target_size):
    """Plots Training patches and Validation square crops."""
    print("\n🎨 Generating Visualizations...")
    fig, axes = plt.subplots(2, 6, figsize=(20, 8))
    
    # Plot 3 Train Forged, 3 Train Auth
    train_samples = train_meta['forged'][:3] + train_meta['authentic'][:3]
    for i, (fpath, x, y, size, is_forged) in enumerate(train_samples):
        img = cv2.cvtColor(cv2.imread(str(fpath)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        y_e, x_e = min(y+size, h), min(x+size, w)
        p_img = img[y:y_e, x:x_e]
        if p_img.shape[:2] != (size, size):
            p_img = cv2.copyMakeBorder(p_img, 0, size-p_img.shape[0], 0, size-p_img.shape[1], cv2.BORDER_CONSTANT)
        
        axes[0, i].imshow(p_img)
        axes[0, i].set_title(f"TRAIN {'F' if is_forged else 'A'}")
        axes[0, i].axis('off')

    # Plot 6 Validation center crops (3F, 3A)
    val_base = CFG.root_dir / "val"
    val_samples = sorted(list((val_base / "forged").glob("*")))[:3] + sorted(list((val_base / "authentic").glob("*")))[:3]
    for i, fpath in enumerate(val_samples):
        is_f = "forged" in str(fpath)
        img = cv2.cvtColor(cv2.imread(str(fpath)), cv2.COLOR_BGR2RGB)
        mask = None
        if is_f:
            m_p = val_base / "masks" / (fpath.stem + ".npy")
            if m_p.exists():
                mask = np.load(m_p)
                if mask.ndim == 3: mask = mask[0]
        
        v_img, v_mask = process_val_image(img, mask, target_size)
        axes[1, i].imshow(v_img if i%2==0 else (v_mask if v_mask is not None else v_img), cmap='magma' if i%2!=0 else None)
        axes[1, i].set_title(f"VAL {'F' if is_f else 'A'}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig("data_preview.png")
    print("📈 Preview saved as 'data_preview.png'")
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    organize_data()
    train_metadata = scan_and_visualize_train(CFG.root_dir, CFG.patch_size, CFG.target_limit)
    visualize_results(train_metadata, CFG.patch_size)