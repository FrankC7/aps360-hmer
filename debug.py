'''
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Import components from train.py and model.py
from model import HMER_Model
from train import (
    Vocabulary,
    CROHMEDataset,
    HMERCollator,
    EMBED_DIM,
    ENCODER_DIM,
    DECODER_DIM,
    TRAIN_TXT,
    VAL_TXT,
    DATA_DIR,
)

CHECKPOINT_PATH = "./checkpoints/best_hmer_model.pth"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"DEBUGGER: Targeting Device -> {device}")

    # 1. Check if annotation file exists
    if not os.path.exists(TRAIN_TXT):
        print(f"ERROR: Annotation file not found at {TRAIN_TXT}")
        return

    # 2. Build vocab
    vocab = Vocabulary()
    vocab.build_vocab(TRAIN_TXT)

    # 3. Test dataset loading and inspect paths
    print("\n--- STAGE 1: VERIFYING IMAGE PATHS ---")
    dataset = CROHMEDataset(TRAIN_TXT, vocab, data_dir=DATA_DIR, is_training=False)
    print(f"Loaded {len(dataset)} entries from {TRAIN_TXT}")

    missing_count = 0
    bg_colors = []
    
    # Check physical existence of the first 100 images
    for i in range(min(100, len(dataset))):
        img_path, tokens = dataset.samples[i]
        if not os.path.exists(img_path):
            missing_count += 1
            if missing_count <= 5:
                print(f"Missing file sample: {img_path}")
        else:
            # Let's inspect the actual background color of the first valid image
            if len(bg_colors) < 5:
                img = Image.open(img_path).convert("L")
                corners = [
                    img.getpixel((0,0)), 
                    img.getpixel((img.width-1, 0)), 
                    img.getpixel((0, img.height-1)), 
                    img.getpixel((img.width-1, img.height-1))
                ]
                avg_corner = sum(corners) / len(corners)
                bg_colors.append((img_path, avg_corner))

    print(f"\nPath Check Results:")
    print(f"  - Missing files in first 100 samples: {missing_count}/100")
    if missing_count > 0:
        print(f"WARNING: Your dataset is skipping missing files! The model is training on BLANK visual data.")
    else:
        print(f"SUCCESS: File path verification passed!")

    print(f"\nBackground Color Check:")
    for path, val in bg_colors:
        color_type = "WHITE (Light background)" if val > 128 else "BLACK (Dark background)"
        print(f"  - {os.path.basename(path)} average corner pixel: {val:.1f} -> Detected as {color_type}")

    # 4. Check outputs qualitative predictions
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\nSkipped Stage 2: Checkpoint file empty at {CHECKPOINT_PATH}")
        return

    print("\n--- STAGE 2: QUALITATIVE PREDICTION CHECK ---")
    model = HMER_Model(
        vocab_size=len(vocab),
        sos_idx=vocab.sos_idx,
        eos_idx=vocab.eos_idx,
        embed_dim=EMBED_DIM,
        encoder_dim=ENCODER_DIM,
        decoder_dim=DECODER_DIM,
    ).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load 5 random samples from validation set
    val_dataset = CROHMEDataset(VAL_TXT, vocab, data_dir=DATA_DIR, is_training=False)
    indices = np.random.choice(len(val_dataset), 5, replace=False)

    for idx in indices:
        img_path, target_tokens = val_dataset.samples[idx]
        if not os.path.exists(img_path):
            continue
            
        # Run inference
        img = Image.open(img_path).convert("L")
        tgt_h, tgt_w = (128, 400)
        img.thumbnail((tgt_w, tgt_h), Image.Resampling.LANCZOS)
        
        # Decide canvas padding based on the real image background color
        corners = [img.getpixel((0,0)), img.getpixel((img.width-1,0))]
        is_dark = (sum(corners)/2) < 128
        pad_color = 0 if is_dark else 255
        
        padded_canvas = Image.new("L", (tgt_w, tgt_h), color=pad_color)
        padded_canvas.paste(img, ((tgt_w - img.width)//2, (tgt_h - img.height)//2))
        
        tensor = transforms.ToTensor()(padded_canvas).unsqueeze(0).to(device)
        tensor = transforms.Normalize(mean=[0.5], std=[0.5])(tensor)

        with torch.no_grad():
            logits, _ = model(tensor, targets=None, max_len=len(target_tokens) + 10, teacher_forcing_ratio=0.0)
            preds = logits.squeeze(0).argmax(dim=-1).cpu().numpy()
            predicted_tokens = vocab.decode(preds)

        print("-" * 60)
        print(f"File Path: {img_path}")
        print(f"Ground Truth: {' '.join(target_tokens)}")
        print(f"Model Predict: {' '.join(predicted_tokens)}")

if __name__ == "__main__":
    main()
'''

# Save this temporarily as check_vocab.py and run it
from train import Vocabulary

vocab = Vocabulary()
vocab.build_vocab("data/crohme2019/crohme2019_train.txt")

# Search for inequality tokens
inequality_tokens = [
    tok for tok in vocab.idx_to_token 
    if any(x in tok.lower() for x in ['less', 'great', 'le', 'leq', 'lt', 'gt', '<', '>'])
]
print("Found Inequality Tokens in Dataset Vocab:", inequality_tokens)

# Print out your whole vocabulary to see how characters are represented
print("\nFull Vocabulary:")
print(vocab.idx_to_token)