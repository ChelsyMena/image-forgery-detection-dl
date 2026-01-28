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
    root_dir = base_path / "data_fullpatches"
    
    patch_size = 512
    target_limit = 20000  # Cap for patch scanning
    batch_size = 8
    device = "mps" if torch.backends.mps.is_available() else "cpu"

# --- 1. DATA ORGANIZATION ---
def organize_data():
    """Splits raw images into train/val folders and maps .npy masks."""
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

# --- 2. PATCH SCANNER (UNIFIED FOR TRAIN & VAL) ---
def scan_patches(root_dir, split, patch_size, target_limit=None):
    """
    Scans a split folder using exact stride and threshold logic.
    
    Args:
        root_dir: Base directory containing data
        split: 'train' or 'val'
        patch_size: Size of patches to extract
        target_limit: Maximum patches per category (None = unlimited)
    """
    print(f"\n🔍 Scanning {split.upper()} patches (Size: {patch_size})...")
    split_base = root_dir / split
    patch_metadata = {'authentic': [], 'forged': []}

    for is_forged in [True, False]:
        category = 'forged' if is_forged else 'authentic'
        folder = split_base / category
        files = sorted([f for f in folder.glob('*') if not f.name.startswith('.')])
        
        count = 0
        for fpath in tqdm(files, desc=f"Scanning {split} {category}"):
            if target_limit and count >= target_limit:
                break
            try:
                if is_forged:
                    m_path = split_base / "masks" / (fpath.stem + ".npy")
                    mask = np.load(m_path)
                    if mask.ndim == 3:
                        mask = mask[0]
                    h, w = mask.shape
                else:
                    img_info = cv2.imread(str(fpath))
                    if img_info is None:
                        continue
                    h, w = img_info.shape[:2]

                stride = patch_size if is_forged else patch_size * 2
                
                for y in range(0, h, stride):
                    for x in range(0, w, stride):
                        if is_forged:
                            m_patch = mask[y:y+patch_size, x:x+patch_size]
                            if np.sum(m_patch > 0) < 50:
                                continue
                        
                        patch_metadata[category].append((fpath, x, y, patch_size, is_forged))
                        count += 1
                        if target_limit and count >= target_limit:
                            break
                    if target_limit and count >= target_limit:
                        break
            except:
                continue

    print(f"📊 {split.upper()} COUNTS: Forged: {len(patch_metadata['forged'])}, Auth: {len(patch_metadata['authentic'])}")
    return patch_metadata

# --- 3. WRAPPER FUNCTIONS ---
def scan_and_visualize_train(root_dir, patch_size, target_limit):
    """Scans TRAIN folder with limit."""
    return scan_patches(root_dir, 'train', patch_size, target_limit)

def scan_and_visualize_val(root_dir, patch_size):
    """Scans VAL folder without limit."""
    return scan_patches(root_dir, 'val', patch_size, target_limit=None)

# --- 4. VALIDATION PROCESSING LOGIC (NOW JUST FOR INDIVIDUAL PATCH EXTRACTION) ---
def process_val_image(img, mask=None, target_size=512):
    """
    Simple resize to square for backward compatibility.
    NOTE: This is now mostly unused since we're using patches.
    """
    img_resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST) if mask is not None else None
    return img_resized, mask_resized


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    organize_data()
    train_metadata = scan_and_visualize_train(CFG.root_dir, CFG.patch_size, CFG.target_limit)
    val_metadata = scan_and_visualize_val(CFG.root_dir, CFG.patch_size)