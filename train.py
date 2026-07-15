import os
import random
import editdistance
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from tqdm import tqdm

from model import HMER_Model

# =====================================================================
# HYPERPARAMETERS & CONFIGURATIONS
# =====================================================================
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 3e-4  # Optimized for AdamW parameter convergence
CLIP_NORM = 2.0  # Tighter, safer gradient clipper boundaries
SEED = 42

# Network dimensions
EMBED_DIM = 256
ENCODER_DIM = 512
DECODER_DIM = 512

# File paths & Folder directories
TRAIN_TXT = "data/crohme2019/crohme2019_train.txt"
VAL_TXT = "data/crohme2019/crohme2019_valid.txt"
DATA_DIR = "data"
SAVE_DIR = "./checkpoints"

NUM_WORKERS = 2

# Tokenizer Constants
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2


# =====================================================================
# Explicit Weight Initialization Logic (MPS Compatible)
# =====================================================================
def init_weights(m):
    """Initializes models with healthy variance targets prior to batch processing.

    Replaced orthogonal_ bounds with xavier_uniform_ to ensure full Apple
    Silicon MPS stability without falling back to CPU matrices.
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.LSTMCell):
        nn.init.xavier_uniform_(m.weight_ih)
        nn.init.xavier_uniform_(m.weight_hh)
        nn.init.constant_(m.bias_ih, 0.0)
        nn.init.constant_(m.bias_hh, 0.0)


# =====================================================================
# 1. Custom Vocabulary Class
# =====================================================================
class Vocabulary:

    def __init__(self):
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"

        self.idx_to_token = [
            self.pad_token,
            self.sos_token,
            self.eos_token,
            self.unk_token,
        ]
        self.token_to_idx = {
            tok: idx for idx, tok in enumerate(self.idx_to_token)
        }

    @property
    def pad_idx(self):
        return self.token_to_idx[self.pad_token]

    @property
    def sos_idx(self):
        return self.token_to_idx[self.sos_token]

    @property
    def eos_idx(self):
        return self.token_to_idx[self.eos_token]

    @property
    def unk_idx(self):
        return self.token_to_idx[self.unk_token]

    def add_token(self, token):
        if token not in self.token_to_idx:
            self.token_to_idx[token] = len(self.idx_to_token)
            self.idx_to_token.append(token)

    def build_vocab(self, txt_path):
        print(f"Building Vocabulary mapping from: {txt_path} ...")
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "\t" not in line:
                    continue
                _, label_str = line.split("\t", 1)
                for token in label_str.split():
                    self.add_token(token)
        print(f"Vocabulary successfully built! Total Size: {len(self)}")

    def numericalize(self, tokens):
        indices = [self.sos_idx]
        for tok in tokens:
            indices.append(self.token_to_idx.get(tok, self.unk_idx))
        indices.append(self.eos_idx)
        return indices

    def decode(self, indices):
        tokens = []
        for idx in indices:
            token = self.idx_to_token[idx]
            if token == self.eos_token:
                break
            if token not in [self.pad_token, self.sos_token]:
                tokens.append(token)
        return tokens

    def __len__(self):
        return len(self.idx_to_token)


# =====================================================================
# 2. Dataset Reader with safe scale normalization
# =====================================================================
class CROHMEDataset(Dataset):

    def __init__(
        self,
        txt_path,
        vocab,
        data_dir="data",
        target_size=(128, 400),
        is_training=True,
    ):
        self.vocab = vocab
        self.data_dir = data_dir
        self.target_size = target_size
        self.is_training = is_training
        self.samples = []

        if is_training:
            self.transform = transforms.Compose(
                [
                    transforms.RandomAffine(
                        degrees=3,
                        translate=(0.02, 0.02),
                        scale=(0.98, 1.02),
                        fill=255,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ]
            )

        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "\t" not in line:
                    continue
                inkml_rel_path, label_str = line.split("\t", 1)
                png_rel_path = inkml_rel_path.replace(".inkml", ".png")
                full_image_path = os.path.join(self.data_dir, png_rel_path)
                tokens = label_str.split()
                self.samples.append((full_image_path, tokens))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, tokens = self.samples[idx]

        try:
            image = Image.open(img_path).convert("L")
        except FileNotFoundError:
            image = Image.new("L", self.target_size, color=255)

        tgt_h, tgt_w = self.target_size
        image.thumbnail((tgt_w - 4, tgt_h - 4), Image.Resampling.LANCZOS)

        # Apply robust scale adjustments during training on the raw PIL canvas
        if self.is_training:
            scale_factor = random.uniform(0.8, 1.2)
            max_w_scale = (tgt_w - 4) / image.width
            max_h_scale = (tgt_h - 4) / image.height
            safe_scale = min(scale_factor, max_w_scale, max_h_scale)

            new_w = max(10, int(image.width * safe_scale))
            new_h = max(10, int(image.height * safe_scale))
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        padded_canvas = Image.new("L", (tgt_w, tgt_h), color=255)
        offset_x = (tgt_w - image.width) // 2
        offset_y = (tgt_h - image.height) // 2
        padded_canvas.paste(image, (offset_x, offset_y))

        image_tensor = self.transform(padded_canvas)
        target_indices = self.vocab.numericalize(tokens)
        target_tensor = torch.tensor(target_indices, dtype=torch.long)

        return image_tensor, target_tensor


class HMERCollator:

    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images, targets = zip(*batch)
        images = torch.stack(images, dim=0)
        padded_targets = nn.utils.rnn.pad_sequence(
            targets, batch_first=True, padding_value=self.pad_idx
        )
        return images, padded_targets


# =====================================================================
# 3. Model Evaluation Details
# =====================================================================
def evaluate_epoch_metrics(logits, targets, vocab):
    preds = logits.argmax(dim=-1).cpu().numpy()
    gts = targets[:, 1:].cpu().numpy()

    total_dist = 0
    total_len = 0
    exact_match = 0
    num_samples = len(preds)

    for pred_seq, gt_seq in zip(preds, gts):
        p_clean = vocab.decode(pred_seq)
        gt_clean = vocab.decode(gt_seq)

        dist = editdistance.eval(p_clean, gt_clean)
        total_dist += dist
        total_len += len(gt_clean) if len(gt_clean) > 0 else 1

        if p_clean == gt_clean:
            exact_match += 1

    edit_err_ratio = total_dist / total_len
    exact_match_ratio = exact_match / num_samples

    return edit_err_ratio, exact_match_ratio


# =====================================================================
# 4. Training loop steps routines
# =====================================================================
def train_one_epoch(
    model, loader, optimizer, criterion, device, tf_ratio, clip_norm=2.0
):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc="Training", leave=False, ncols=100)
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits, _ = model(images, targets, teacher_forcing_ratio=tf_ratio)

        out_vocab_dim = logits.size(-1)
        loss_logits = logits.contiguous().view(-1, out_vocab_dim)
        loss_targets = targets[:, 1:].contiguous().view(-1)

        loss = criterion(loss_logits, loss_targets)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(Loss=loss.item())

    return total_loss / len(loader)


def validate_one_epoch(model, loader, criterion, vocab, device):
    model.eval()
    total_loss = 0.0
    total_edit_err = 0.0
    total_exact_match = 0.0

    with torch.no_grad():
        for images, targets in tqdm(
            loader, desc="Validating", leave=False, ncols=100
        ):
            images = images.to(device)
            targets = targets.to(device)

            logits, _ = model(images, targets, teacher_forcing_ratio=0.0)

            out_vocab_dim = logits.size(-1)
            loss_logits = logits.contiguous().view(-1, out_vocab_dim)
            loss_targets = targets[:, 1:].contiguous().view(-1)

            loss = criterion(loss_logits, loss_targets)
            total_loss += loss.item()

            edit_err, exact_acc = evaluate_epoch_metrics(logits, targets, vocab)
            total_edit_err += edit_err
            total_exact_match += exact_acc

    num_batches = len(loader)
    return (
        total_loss / num_batches,
        total_edit_err / num_batches,
        total_exact_match / num_batches,
    )


# =====================================================================
# 5. Main Execution Entrypoint
# =====================================================================
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    else:
        device = torch.device("cpu")

    print(f"System configured computing target device: {device}")

    vocab = Vocabulary()
    vocab.build_vocab(TRAIN_TXT)

    collator = HMERCollator(vocab.pad_idx)

    print("Pre-loading Datasets configuration...")
    train_dataset = CROHMEDataset(
        TRAIN_TXT, vocab, data_dir=DATA_DIR, is_training=True
    )
    val_dataset = CROHMEDataset(
        VAL_TXT, vocab, data_dir=DATA_DIR, is_training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=NUM_WORKERS,
    )

    model = HMER_Model(
        vocab_size=len(vocab),
        sos_idx=vocab.sos_idx,
        eos_idx=vocab.eos_idx,
        pad_idx=vocab.pad_idx,
        embed_dim=EMBED_DIM,
        encoder_dim=ENCODER_DIM,
        decoder_dim=DECODER_DIM,
    ).to(device)

    # Apply stable weight initialization
    model.apply(init_weights)

    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab.pad_idx, label_smoothing=0.1
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    best_val_loss = float("inf")
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("\nStarting Updated and Evaluated HMER Coverage Network Training!")
    for epoch in range(1, EPOCHS + 1):
        tf_ratio = max(0.20, 1.0 - (epoch - 1) / EPOCHS)
        print(
            f"\n--- Epoch {epoch:02d}/{EPOCHS:02d} (Teacher Forcing: {tf_ratio:.2%}) ---"
        )

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            tf_ratio,
            CLIP_NORM,
        )
        val_loss, val_edit, val_exact = validate_one_epoch(
            model, val_loader, criterion, vocab, device
        )

        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{EPOCHS:02d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Character Error Rate: {val_edit:.4%} | Val Exact Match: {val_exact:.4%}"
        )

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(SAVE_DIR, "best_hmer_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "vocab_size": len(vocab),
                    "val_loss": val_loss,
                },
                best_checkpoint_path,
            )
            print(
                f"Successfully saved new best model to {best_checkpoint_path}\n"
            )


if __name__ == "__main__":
    main()