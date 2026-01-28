import os
import torch
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize
import gc
import time
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
import numba
from numba import jit, njit
import numpy.typing as npt

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, Subset
import cv2
import numpy as np
from pathlib import Path


class UNetForgeryDataset(Dataset):
    def __init__(self, root_dir, img_size=512, augment=False):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.augment = augment
        
        # Load paths
        self.authentic_paths = list((self.root_dir / "train_images" / "authentic").glob("*.png"))
        self.forged_paths = list((self.root_dir / "train_images" / "forged").glob("*.png"))
        self.all_paths = self.authentic_paths + self.forged_paths
        
        # ImageNet normalization (applied on 0-1 range tensors)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
        
        # Augmentation pipelines
        if augment:
            # Color augmentation (only for images)
            self.color_transform = T.RandomApply([
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
            ], p=0.3)
        else:
            self.color_transform = None
        
        # Build Metadata for curriculum (if needed)
        self.curriculum_meta = []
        print(f"Scanning {len(self.all_paths)} images...")
        
        forged_count = 0
        forged_with_masks = 0
        mask_load_errors = 0
        
        for idx, img_path in enumerate(self.all_paths):
            area = 0
            if "forged" in str(img_path):
                forged_count += 1
                mask_path = self.root_dir / "train_masks" / (img_path.stem + ".npy")
                
                if mask_path.exists():
                    try:
                        mask = np.load(mask_path).squeeze()
                        if mask.ndim > 2:
                            mask = mask[:, :, 0]
                        
                        # Binary masks: use > 0 (since masks are 0 or 1)
                        area = np.sum(mask > 0)
                        
                        if area > 0:
                            forged_with_masks += 1
                    except Exception as e:
                        mask_load_errors += 1
                        if mask_load_errors <= 3:  # Only print first few errors
                            print(f"⚠️  Error loading mask for {img_path.name}: {e}")
                else:
                    if forged_count <= 3:  # Only print first few missing masks
                        print(f"⚠️  Mask not found: {mask_path}")
                        
            self.curriculum_meta.append((idx, 0, area))
        
        print(f"✅ Scan complete:")
        print(f"   Total forged images: {forged_count}")
        print(f"   Forged with valid masks: {forged_with_masks}")
        print(f"   Mask load errors: {mask_load_errors}")
        print(f"   Total authentic: {len(self.authentic_paths)}")

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        img_path = self.all_paths[idx]
        
        # 1. Load Image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        # 2. Load Mask
        mask = np.zeros((h, w), dtype=np.float32)
        is_forged = "forged" in str(img_path)
        
        if is_forged:
            mask_path = self.root_dir / "train_masks" / (img_path.stem + ".npy")
            if mask_path.exists():
                m = np.load(mask_path)
                # --- FIX: Handle Channel-First (C, H, W) ---
                if m.ndim > 2: 
                    # Your manual check confirmed m[0] is the correct mask
                    m = m[0] 
                
                # Resize to match image if needed
                mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

        # 3. Hit-Confirm Crop (Safe Logic)
        if is_forged and np.any(mask > 0):
            # Get valid pixels
            coords = np.argwhere(mask > 0)
            
            # Pick Random Anchor
            anchor_y, anchor_x = coords[np.random.randint(len(coords))]
            
            # Jitter around anchor
            offset_y = np.random.randint(0, self.img_size)
            offset_x = np.random.randint(0, self.img_size)
            
            start_y = max(0, min(h - self.img_size, anchor_y - offset_y))
            start_x = max(0, min(w - self.img_size, anchor_x - offset_x))
            
        else:
            # Authentic / Empty: Random Crop
            start_y = np.random.randint(0, max(1, h - self.img_size))
            start_x = np.random.randint(0, max(1, w - self.img_size))

        # 4. Perform Crop
        img_crop = img[start_y:start_y+self.img_size, start_x:start_x+self.img_size]
        mask_crop = mask[start_y:start_y+self.img_size, start_x:start_x+self.img_size]

        # 5. Padding
        if img_crop.shape[0] < self.img_size or img_crop.shape[1] < self.img_size:
            img_crop = cv2.copyMakeBorder(img_crop, 0, self.img_size-img_crop.shape[0], 0, self.img_size-img_crop.shape[1], cv2.BORDER_CONSTANT)
            mask_crop = cv2.copyMakeBorder(mask_crop, 0, self.img_size-mask_crop.shape[0], 0, self.img_size-mask_crop.shape[1], cv2.BORDER_CONSTANT)

        # 6. To Tensor
        img_tensor = torch.from_numpy(img_crop.astype(np.float32) / 255.0).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_crop > 0).float().unsqueeze(0)

        return img_tensor, mask_tensor

    def get_loaders(self, batch_size=8, val_split=0.2, device='cpu', num_workers=2):
        """
        Returns two BalancedComboLoader objects (Train, Val).
        No normalization in loader - already done in __getitem__.
        """
        # Prepare indices
        all_meta = self.curriculum_meta
        forged_indices = [x[0] for x in all_meta if x[2] > 0]
        auth_indices = [x[0] for x in all_meta if x[2] == 0]
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total images: {len(self.all_paths)}")
        print(f"   Forged images: {len(forged_indices)}")
        print(f"   Authentic images: {len(auth_indices)}")
        
        # Safety check
        if len(forged_indices) == 0:
            raise ValueError("No forged images found! Check your data directory structure.")
        if len(auth_indices) == 0:
            raise ValueError("No authentic images found! Check your data directory structure.")

        # Random split
        import random
        random.seed(42)
        random.shuffle(forged_indices)
        random.shuffle(auth_indices)

        # Ensure at least 1 sample in validation if possible
        n_val_f = max(1, int(len(forged_indices) * val_split)) if len(forged_indices) > 1 else 0
        n_val_a = max(1, int(len(auth_indices) * val_split)) if len(auth_indices) > 1 else 0
        
        # If very small dataset, disable validation split
        if len(forged_indices) < 5 or len(auth_indices) < 5:
            print("⚠️  Warning: Very small dataset detected. Using minimal validation split.")
            n_val_f = 1 if len(forged_indices) > 1 else 0
            n_val_a = 1 if len(auth_indices) > 1 else 0

        train_f_idx = forged_indices[n_val_f:]
        val_f_idx = forged_indices[:n_val_f]
        
        train_a_idx = auth_indices[n_val_a:]
        val_a_idx = auth_indices[:n_val_a]
        
        print(f"   Train forged: {len(train_f_idx)}")
        print(f"   Train authentic: {len(train_a_idx)}")
        print(f"   Val forged: {len(val_f_idx)}")
        print(f"   Val authentic: {len(val_a_idx)}")
        
        # Final safety check
        if len(train_f_idx) == 0 or len(train_a_idx) == 0:
            raise ValueError("Training split is empty! Need at least 2 images per class.")
        if len(val_f_idx) == 0 or len(val_a_idx) == 0:
            print("⚠️  Warning: Validation split is empty. Training without validation.")
            # Use a small subset of training data for validation
            val_f_idx = train_f_idx[:1]
            val_a_idx = train_a_idx[:1]

        # Optional: Sort training forgeries (smallest to largest - true curriculum)
        # area_lookup = {x[0]: x[2] for x in all_meta}
        # train_f_idx.sort(key=lambda idx: area_lookup[idx])  # ascending = easy to hard

        # Create sub-loaders
        half_batch = batch_size // 2
        
        # Enable augmentation for training, disable for validation
        train_dataset = self
        val_dataset = UNetForgeryDataset(
            self.root_dir, 
            img_size=self.img_size, 
            augment=False  # No augmentation for validation
        )
        val_dataset.all_paths = train_dataset.all_paths
        val_dataset.curriculum_meta = train_dataset.curriculum_meta
        
        dl_train_f = DataLoader(
            Subset(train_dataset, train_f_idx), 
            batch_size=half_batch, 
            shuffle=True,  # Shuffle for better training
            num_workers=num_workers, 
            pin_memory=True, 
            drop_last=True
        )
        dl_train_a = DataLoader(
            Subset(train_dataset, train_a_idx), 
            batch_size=half_batch, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=True, 
            drop_last=True
        )
        
        dl_val_f = DataLoader(
            Subset(val_dataset, val_f_idx), 
            batch_size=half_batch, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=True
        )
        dl_val_a = DataLoader(
            Subset(val_dataset, val_a_idx), 
            batch_size=half_batch, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=True
        )

        # Wrap in BalancedComboLoader (no normalization needed now)
        train_loader = BalancedComboLoader(dl_train_f, dl_train_a, device)
        val_loader = BalancedComboLoader(dl_val_f, dl_val_a, device)
        
        print(f"📦 Data Prepared: {len(train_loader)} Train Batches | {len(val_loader)} Val Batches")
        
        return train_loader, val_loader


class BalancedComboLoader:
    """
    Wraps two DataLoaders (Forged & Authentic) and serves balanced 50/50 batches.
    NO normalization applied here - already done in dataset.
    """
    def __init__(self, loader_forged, loader_auth, device):
        self.loader_forged = loader_forged
        self.loader_auth = loader_auth
        self.device = device

    def __iter__(self):
        iterator = zip(self.loader_forged, self.loader_auth)
        
        for (img_f, mask_f), (img_a, mask_a) in iterator:
            # Concatenate to make batch of 8
            imgs = torch.cat([img_f, img_a])
            masks = torch.cat([mask_f, mask_a])
            
            # Move to device (data is already normalized)
            imgs = imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            
            yield imgs, masks

    def __len__(self):
        return min(len(self.loader_forged), len(self.loader_auth))


# --- SIMPLIFIED TRAIN FUNCTION ---

def train_model(
    model, dataset, device,
    epochs=30, batch_size=8, lr=1e-4,
    model_name="ManTraNet", val_split=0.2,
    output_dir=".",
    criterion=None  # <--- NEW ARGUMENT (Optional)
):
    import torch.optim as optim
    from tqdm import tqdm
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    # 1. Setup Output
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"📂 Output Directory: {save_path.resolve()}")

    # 2. Loaders
    train_loader, val_loader = dataset.get_loaders(batch_size, val_split, device)

    # 3. Setup Loss (The Flexible Logic)
    if criterion is None:
        print("ℹ️ No criterion provided. Using Default: Weighted BCEWithLogitsLoss")
        pos_weight = torch.tensor([10.0], device=device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        print(f"ℹ️ Using Custom Criterion: {type(criterion).__name__}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()
    
    history = {'train_batch_loss': [], 'val_epoch_loss': []}

    print(f"🚀 Training '{model_name}' on {device}...")

    # --- Training Loop (Same as before) ---
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for imgs, masks in pbar:
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                logits = model(imgs)
                # The criterion is now whatever you passed in!
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            history['train_batch_loss'].append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                with torch.cuda.amp.autocast():
                    val_loss += criterion(model(imgs), masks).item()

        avg_val_loss = val_loss / max(len(val_loader), 1)
        history['val_epoch_loss'].append(avg_val_loss)
        print(f"✅ Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f}")

        # --- Saving (Same as before) ---
        df_batch = pd.DataFrame({'batch_loss': history['train_batch_loss']})
        df_batch.to_csv(save_path / f"{model_name}_batch_losses.csv", index_label='step')
        
        df_epoch = pd.DataFrame({'val_loss': history['val_epoch_loss']})
        df_epoch.to_csv(save_path / f"{model_name}_epoch_metrics.csv", index_label='epoch')

        # Plotting code (Same as before)...
        # (Abbreviated for brevity, paste the previous plotting code here)
        plt.figure(figsize=(12, 6))
        plt.plot(history['train_batch_loss'], color='gray', alpha=0.3)
        plt.plot(history['val_epoch_loss'], color='red', marker='o')
        plt.savefig(save_path / f"{model_name}_training_plot.png")
        plt.close()

        torch.save(model.state_dict(), save_path / f"{model_name}_latest.pt")
        np.save(save_path / f"{model_name}_history.npy", history)

    return history

# --- 4. VISUALIZATION & PLOTTING ---

def get_binary_mask(model, img_tensor, device, threshold=0.5):
    """Helper to get a binary mask from model output."""
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor.unsqueeze(0).to(device))
        probs = torch.sigmoid(logits)
        return (probs > threshold).float().cpu().numpy()[0, 0]

def visualize_prediction(model, dataset, device, index=0):
    """Visualizes a single prediction from the dataset."""
    img_tensor, mask_tensor = dataset[index]
    pred_mask = get_binary_mask(model, img_tensor, device)
    
    img_np = img_tensor.permute(1, 2, 0).numpy()
    mask_np = mask_tensor.squeeze().numpy()
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.title("Original"); plt.imshow(img_np)
    plt.subplot(1, 3, 2); plt.title("True Mask"); plt.imshow(mask_np, cmap='gray')
    plt.subplot(1, 3, 3); plt.title("Prediction"); plt.imshow(pred_mask, cmap='gray')
    plt.show()

def visualize_failures(model, dataset, device, threshold=0.5):
    """Visualizes specific failure cases (Small, Large, Complex)."""
    # Filter for interesting samples
    dtype = [('index', int), ('zones', int), ('area', int)]
    meta_arr = np.array(dataset.curriculum_meta, dtype=dtype)
    indices = []
    
    small = meta_arr[(meta_arr['area'] > 0) & (meta_arr['area'] < 2000)]
    large = meta_arr[meta_arr['area'] > 5000]
    
    if len(small) > 0: indices.append(small[0]['index'])
    if len(large) > 0: indices.append(large[0]['index'])
    if not indices: indices = [0, 1]

    fig, axes = plt.subplots(len(indices), 4, figsize=(16, 4 * len(indices)), squeeze=False)
    
    for row_idx, idx in enumerate(indices):
        img_tensor, mask_tensor = dataset[idx]
        pred_binary = get_binary_mask(model, img_tensor, device, threshold)
        
        img_disp = img_tensor.permute(1, 2, 0).numpy()
        mask_disp = mask_tensor.numpy()[0]
        error_map = mask_disp - pred_binary
        
        ax_row = axes[row_idx]
        ax_row[0].imshow(img_disp); ax_row[0].set_title(f"ID {idx}")
        ax_row[1].imshow(mask_disp, cmap='gray'); ax_row[1].set_title("Ground Truth")
        ax_row[2].imshow(pred_binary, cmap='gray'); ax_row[2].set_title("Prediction")
        ax_row[3].imshow(error_map, cmap='bwr', vmin=-1, vmax=1); ax_row[3].set_title("Error (Red=Missed)")
        for ax in ax_row: ax.axis('off')
    plt.tight_layout()
    plt.show()

def analyze_predictions(model, dataset, device, mode="largest_area", n_samples=3, threshold=0.5):
    """Analyzes dataset based on area or IoU success."""
    dtype = [('index', int), ('zones', int), ('area', int)]
    meta_arr = np.array(dataset.curriculum_meta, dtype=dtype)
    forged_candidates = meta_arr[meta_arr['area'] > 0]
    
    results = []
    print(f"Scanning {len(forged_candidates)} candidates for mode '{mode}'...")
    
    for candidate in tqdm(forged_candidates):
        idx = candidate['index']
        img_tensor, mask_tensor = dataset[idx]
        pred_binary = get_binary_mask(model, img_tensor, device, threshold)
        gt_mask = mask_tensor.numpy()[0]
        
        iou = calculate_iou(pred_binary, gt_mask)
        results.append((idx, iou, candidate['area'], pred_binary, gt_mask))

    # Sorting Logic
    if mode == "largest_area":
        results.sort(key=lambda x: x[2], reverse=True)
    elif mode == "big_success":
        results = [r for r in results if r[2] > 5000]
        results.sort(key=lambda x: x[1], reverse=True)
    elif mode == "worst_failure":
        results.sort(key=lambda x: x[1], reverse=False)

    indices_to_show = results[:n_samples]
    
    fig, axes = plt.subplots(len(indices_to_show), 4, figsize=(16, 4 * len(indices_to_show)), squeeze=False)
    for row_idx, (idx, iou, area, pred_bin, gt_mask) in enumerate(indices_to_show):
        img_tensor, _ = dataset[idx]
        img_disp = img_tensor.permute(1, 2, 0).numpy()
        error_map = gt_mask - pred_bin
        
        ax_row = axes[row_idx]
        ax_row[0].imshow(img_disp); ax_row[0].set_title(f"Area={area}")
        ax_row[1].imshow(gt_mask, cmap='gray'); ax_row[1].set_title("GT")
        ax_row[2].imshow(pred_bin, cmap='gray'); ax_row[2].set_title(f"IoU={iou:.3f}")
        ax_row[3].imshow(error_map, cmap='bwr', vmin=-1, vmax=1); ax_row[3].set_title("Error")
        for ax in ax_row: ax.axis('off')
    plt.tight_layout()
    plt.show()

# --- 5. TRAINING UTILITIES (Loss, Time) ---

def save_loss_plot(losses, filename="loss_curve.png"):
    """Saves the training loss curve to a file."""
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Training Loss")
    plt.title("Training Loss Curve")
    plt.xlabel("Steps/Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"📉 Loss plot saved to {filename}")

class Timer:
    """Simple context manager to time code execution."""
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print(f"⏱️ Time elapsed: {self.interval:.2f} seconds")



# --- APPEND TO utils.py ---

def visualize_metrics_summary(model, dataset, device, n_samples=50):
    """
    Computes and plots Confusion Matrix, IoU, and F1 distribution.
    """
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    model.eval()
    all_preds = []
    all_gts = []
    ious = []
    
    print(f"📊 Computing metrics on {n_samples} samples...")
    
    # Randomly sample the dataset
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    
    with torch.no_grad():
        for idx in indices:
            img_tensor, mask_tensor = dataset[idx]
            # Get binary prediction
            logits = model(img_tensor.unsqueeze(0).to(device))
            pred = (torch.sigmoid(logits) > 0.5).float().cpu().numpy().flatten()
            gt = mask_tensor.numpy().flatten()
            
            all_preds.extend(pred)
            all_gts.extend(gt)
            
            # Calculate IoU for this image
            intersection = np.logical_and(pred, gt).sum()
            union = np.logical_or(pred, gt).sum()
            if union > 0:
                ious.append(intersection/union)
            else:
                ious.append(1.0) # Both empty = Perfect match
                
    # 1. Confusion Matrix
    cm = confusion_matrix(all_gts, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    # 2. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    # Confusion Matrix Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
                xticklabels=['Background', 'Forgery'], yticklabels=['Background', 'Forgery'])
    axes[0].set_title(f"Pixel-wise Confusion Matrix\n(Sample of {n_samples} imgs)")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")
    
    # IoU Histogram
    axes[1].hist(ious, bins=20, color='orange', edgecolor='black')
    axes[1].set_title(f"IoU Distribution (Mean: {np.mean(ious):.3f})")
    axes[1].set_xlabel("IoU Score")
    
    # Metrics Text
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    text_str = f"Global Metrics:\n\nF1 Score: {f1:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\n\nTP: {tp}\nFP: {fp}"
    axes[2].text(0.1, 0.5, text_str, fontsize=14, transform=axes[2].transAxes)
    axes[2].axis('off')
    axes[2].set_title("Summary Stats")
    
    plt.tight_layout()
    plt.show()


    # --- 2. EVALUATION METRICS (RLE, F1, Dice, IoU) ---

class ParticipantVisibleError(Exception):
    pass

@numba.jit(nopython=True)
def _rle_encode_jit(x: npt.NDArray, fg_val: int = 1) -> list[int]:
    """Numba-jitted RLE encoder."""
    dots = np.where(x.T.flatten() == fg_val)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def rle_encode(masks: list[npt.NDArray], fg_val: int = 1) -> str:
    """Adapted from contrails RLE."""
    return ';'.join([json.dumps(_rle_encode_jit(x, fg_val)) for x in masks])

@numba.njit
def _rle_decode_jit(mask_rle: npt.NDArray, height: int, width: int) -> npt.NDArray:
    """Numba-jitted RLE decoder."""
    if len(mask_rle) % 2 != 0:
        raise ValueError('One or more rows has an odd number of values.')
    starts, lengths = mask_rle[0::2], mask_rle[1::2]
    starts -= 1
    ends = starts + lengths
    for i in range(len(starts) - 1):
        if ends[i] > starts[i + 1]:
            raise ValueError('Pixels must not be overlapping.')
    img = np.zeros(height * width, dtype=np.bool_)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img

def rle_decode(mask_rle: str, shape: tuple[int, int]) -> npt.NDArray:
    """Decodes RLE string to numpy array."""
    mask_rle = json.loads(mask_rle)
    mask_rle = np.asarray(mask_rle, dtype=np.int32)
    try:
        return _rle_decode_jit(mask_rle, shape[0], shape[1]).reshape(shape, order='F')
    except ValueError as e:
        raise ParticipantVisibleError(str(e)) from e

def calculate_iou(pred_mask, gt_mask):
    """Calculates Intersection over Union (IoU)."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def calculate_dice(pred_mask, gt_mask, smooth=1e-6):
    """Calculates Dice Coefficient."""
    intersection = (pred_mask * gt_mask).sum()
    return (2. * intersection + smooth) / (pred_mask.sum() + gt_mask.sum() + smooth)

def get_confusion_matrix_stats(pred_mask, gt_mask):
    """Returns TP, FP, TN, FN for a binary mask."""
    pred_flat = pred_mask.flatten().astype(bool)
    gt_flat = gt_mask.flatten().astype(bool)
    
    tp = np.logical_and(pred_flat, gt_flat).sum()
    tn = np.logical_and(~pred_flat, ~gt_flat).sum()
    fp = np.logical_and(pred_flat, ~gt_flat).sum()
    fn = np.logical_and(~pred_flat, gt_flat).sum()
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}

def calculate_f1_score(pred_mask: npt.NDArray, gt_mask: npt.NDArray):
    """Standard F1 Score."""
    stats = get_confusion_matrix_stats(pred_mask, gt_mask)
    tp, fp, fn = stats["TP"], stats["FP"], stats["FN"]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if (precision + recall) > 0:
        return 2 * (precision * recall) / (precision + recall)
    else:
        return 0

def calculate_f1_matrix(pred_masks: list[npt.NDArray], gt_masks: list[npt.NDArray]):
    """Helper for Optimal F1."""
    num_instances_pred = len(pred_masks)
    num_instances_gt = len(gt_masks)
    f1_matrix = np.zeros((num_instances_pred, num_instances_gt))

    for i in range(num_instances_pred):
        for j in range(num_instances_gt):
            f1_matrix[i, j] = calculate_f1_score(pred_mask=pred_masks[i], gt_mask=gt_masks[j])

    if f1_matrix.shape[0] < len(gt_masks):
        f1_matrix = np.vstack((f1_matrix, np.zeros((len(gt_masks) - len(f1_matrix), num_instances_gt))))
    return f1_matrix

def oF1_score(pred_masks: list[npt.NDArray], gt_masks: list[npt.NDArray]):
    """Calculates Optimal F1 using Hungarian Algorithm."""
    f1_matrix = calculate_f1_matrix(pred_masks, gt_masks)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-f1_matrix)
    excess_predictions_penalty = len(gt_masks) / max(len(pred_masks), len(gt_masks))
    return np.mean(f1_matrix[row_ind, col_ind]) * excess_predictions_penalty

class VisualizationDataset(Dataset):
    """Simple dataset for visualization - loads full images without patches"""
    def __init__(self, root_dir, img_size=512):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        
        # Get validation images
        val_path = self.root_dir / "data_patches" / "val"
        self.forged_paths = sorted(list((val_path / "forged").glob("*")))
        self.authentic_paths = sorted(list((val_path / "authentic").glob("*")))
        self.all_paths = self.forged_paths + self.authentic_paths
        
        # Build curriculum_meta for compatibility
        self.curriculum_meta = []
        for idx, img_path in enumerate(self.all_paths):
            area = 0
            if "forged" in str(img_path):
                mask_path = val_path / "masks" / (img_path.stem + ".npy")
                if mask_path.exists():
                    mask = np.load(mask_path)
                    if mask.ndim == 3:
                        mask = mask[0]
                    area = np.sum(mask > 0)
            self.curriculum_meta.append((idx, 0, area))
    
    def __len__(self):
        return len(self.all_paths)
    
    def __getitem__(self, idx):
        img_path = self.all_paths[idx]
        is_forged = "forged" in str(img_path)
        
        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = None
        if is_forged:
            mask_path = self.root_dir / "data_patches" / "val" / "masks" / (img_path.stem + ".npy")
            if mask_path.exists():
                mask = np.load(mask_path)
                if mask.ndim == 3:
                    mask = mask[0]
        
        # Use the same preprocessing as validation
        import preprocess
        img_processed, mask_processed = preprocess.process_val_image(
            img, mask, target_size=self.img_size
        )
        
        # Convert to tensor (match training format)
        img_tensor = torch.from_numpy(img_processed.astype(np.float32) / 255.0).permute(2, 0, 1)
        
        if mask_processed is not None:
            mask_tensor = torch.from_numpy(mask_processed > 0).float().unsqueeze(0)
        else:
            mask_tensor = torch.zeros(1, self.img_size, self.img_size)
        
        return img_tensor, mask_tensor