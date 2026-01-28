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
    base_path = Path("/Users/chelsymena/recodai-luc-scientific-image-forgery-detection")
    raw_img_dir = base_path / 'train_images'
    raw_mask_dir = base_path / 'train_masks'
    
    # Processed paths
    root_dir = base_path / "data_resized"
    
    resize_size = 512
    device = "mps" if torch.backends.mps.is_available() else "cpu"

# --- 1. DATA ORGANIZATION ---
def organize_data():
    """Splits raw images into train/val folders and resizes them."""
    print("📂 Organizing data and splitting into Train/Val...")
    for split in ['train', 'val']:
        for cat in ['authentic', 'forged', 'masks']:
            os.makedirs(CFG.root_dir / split / cat, exist_ok=True)

    # Process Authentic
    auth_files = list(Path(CFG.raw_img_dir).glob('authentic/*'))
    a_train, a_val = train_test_split(auth_files, test_size=0.2, random_state=42)
    
    # Process Forged
    forged_files = list(Path(CFG.raw_img_dir).glob('forged/*'))
    f_train, f_val = train_test_split(forged_files, test_size=0.2, random_state=42)

    def distribute_and_resize_files(files, split, is_forged=False):
        for fpath in tqdm(files, desc=f"Processing {split} {'forged' if is_forged else 'authentic'}"):
            dest_cat = 'forged' if is_forged else 'authentic'
            
            # Load, resize, and save image
            img = cv2.imread(str(fpath))
            if img is None:
                continue
            img_resized = cv2.resize(img, (CFG.resize_size, CFG.resize_size), interpolation=cv2.INTER_AREA)
            dest_path = CFG.root_dir / split / dest_cat / fpath.name
            cv2.imwrite(str(dest_path), img_resized)
            
            # Resize and save mask if forged
            if is_forged:
                mask_name = fpath.stem + '.npy'
                m_path = CFG.raw_mask_dir / mask_name
                if m_path.exists():
                    mask = np.load(m_path)
                    if mask.ndim == 3:
                        mask = mask[0]
                    mask_resized = cv2.resize(mask, (CFG.resize_size, CFG.resize_size), interpolation=cv2.INTER_NEAREST)
                    mask_dest = CFG.root_dir / split / 'masks' / mask_name
                    np.save(str(mask_dest), mask_resized)

    distribute_and_resize_files(a_train, 'train')
    distribute_and_resize_files(a_val, 'val')
    distribute_and_resize_files(f_train, 'train', is_forged=True)
    distribute_and_resize_files(f_val, 'val', is_forged=True)
    print("✅ Resizing and Transfer Complete.")

# --- 2. TRAINING DATA SCANNER ---
def scan_and_visualize_train(root_dir, resize_size):
    """Scans TRAIN folder and returns metadata for all images (no patches)."""
    print(f"\n🔍 Scanning Train images (Resized to: {resize_size}x{resize_size})...")
    train_base = root_dir / "train"
    image_metadata = {'authentic': [], 'forged': []}

    for is_forged in [True, False]:
        category = 'forged' if is_forged else 'authentic'
        folder = train_base / category
        files = sorted([f for f in folder.glob('*') if not f.name.startswith('.')])
        
        for fpath in tqdm(files, desc=f"Scanning {category}"):
            try:
                # Each image is now a single sample (no patches)
                image_metadata[category].append((fpath, 0, 0, resize_size, is_forged))
            except:
                continue

    print(f"📊 TRAIN COUNTS: Forged: {len(image_metadata['forged'])}, Auth: {len(image_metadata['authentic'])}")
    return image_metadata

# --- 3. VALIDATION PROCESSING LOGIC ---
def process_val_image(img, mask=None, target_size=512):
    """Simple resize to square."""
    img_resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST) if mask is not None else None
    
    return img_resized, mask_resized

# --- 4. VISUALIZATION ---
def visualize_results(train_meta, target_size):
    """Plots Training and Validation images (all resized)."""
    print("\n🎨 Generating Visualizations...")
    fig, axes = plt.subplots(2, 6, figsize=(20, 8))
    
    # Plot 3 Train Forged, 3 Train Auth
    train_samples = train_meta['forged'][:3] + train_meta['authentic'][:3]
    for i, (fpath, _, _, _, is_forged) in enumerate(train_samples):
        img = cv2.cvtColor(cv2.imread(str(fpath)), cv2.COLOR_BGR2RGB)
        
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"TRAIN {'F' if is_forged else 'A'}")
        axes[0, i].axis('off')

    # Plot 6 Validation images (3F, 3A)
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
    train_metadata = scan_and_visualize_train(CFG.root_dir, CFG.resize_size)
    visualize_results(train_metadata, CFG.resize_size)