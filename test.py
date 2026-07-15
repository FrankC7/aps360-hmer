import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

# Import components directly from your train.py and model.py to keep code DRY
from model import HMER_Model
from train import (
    Vocabulary,
    CROHMEDataset,
    HMERCollator,
    evaluate_epoch_metrics,
    # Import hyperparameters to rebuild the identical network
    EMBED_DIM,
    ENCODER_DIM,
    DECODER_DIM,
    DATA_DIR,
    TRAIN_TXT,
)

# Test-specific Configurations
TEST_TXT = "data/crohme2019/crohme2019_test.txt"
CHECKPOINT_PATH = "./checkpoints/best_hmer_model.pth"
BATCH_SIZE = 16


# =====================================================================
# 1. Single Image Inference Function
# =====================================================================
def predict_latex(model, img_path, vocab, device, target_size=(128, 400)):
    """Predicts a LaTeX sequence for a single grayscale image."""
    model.eval()

    # Apply identical aspect-ratio preserving padding as training
    try:
        image = Image.open(img_path).convert("L")
    except FileNotFoundError:
        print(f"Error: Could not find image at {img_path}")
        return None

    # Centered Aspect-Ratio-Preserving Padding
    tgt_h, tgt_w = target_size
    image.thumbnail((tgt_w, tgt_h), Image.Resampling.LANCZOS)

    padded_canvas = Image.new("L", (tgt_w, tgt_h), color=255)
    offset_x = (tgt_w - image.width) // 2
    offset_y = (tgt_h - image.height) // 2
    padded_canvas.paste(image, (offset_x, offset_y))

    # Transform to tensor
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    image_tensor = transform(padded_canvas).unsqueeze(0).to(device)  # [1, 1, 128, 400]

    with torch.no_grad():
        # Pass through the model (teacher forcing = 0.0)
        # Sequence ceilings at 150 tokens
        logits, _ = model(
            image_tensor, targets=None, max_len=150, teacher_forcing_ratio=0.0
        )

        # Decode the predicted token indices
        predicted_indices = logits.squeeze(0).argmax(dim=-1).cpu().numpy()
        predicted_tokens = vocab.decode(predicted_indices)

    # Rejoin the tokens into a readable string
    return " ".join(predicted_tokens)


# =====================================================================
# 2. Main Test Execution Orchestration
# =====================================================================
def main():
    # Setup Device targets (including Apple MPS acceleration)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Step 1: Re-build Vocabulary (Must use the identical rules as train.py)
    vocab = Vocabulary()
    vocab.build_vocab(TRAIN_TXT)

    # Step 2: Initialize model structure
    model = HMER_Model(
        vocab_size=len(vocab),
        sos_idx=vocab.sos_idx,
        eos_idx=vocab.eos_idx,
        embed_dim=EMBED_DIM,
        encoder_dim=ENCODER_DIM,
        decoder_dim=DECODER_DIM,
    ).to(device)

    # Step 3: Load the trained model checkpoint weights
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Could not find model checkpoint at {CHECKPOINT_PATH}. "
            f"Please ensure your model has finished training at least one epoch."
        )

    print(f"Loading trained weights from: {CHECKPOINT_PATH}.")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Model weights successfully loaded!")

    # =====================================================================
    # BULK TEST SET EVALUATION
    # =====================================================================
    print(f"\nEvaluating performance on test dataset: {TEST_TXT}.")
    test_dataset = CROHMEDataset(TEST_TXT, vocab, data_dir=DATA_DIR)
    collator = HMERCollator(vocab.pad_idx)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
    )

    model.eval()
    total_edit_err = 0.0
    total_exact_match = 0.0

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Testing", leave=False):
            images = images.to(device)
            targets = targets.to(device)

            # Predict (Inference rules)
            logits, _ = model(images, targets=None, max_len=targets.size(1), teacher_forcing_ratio=0.0)

            # Measure performance accuracies
            edit_err, exact_acc = evaluate_epoch_metrics(logits, targets, vocab)
            total_edit_err += edit_err
            total_exact_match += exact_acc

    num_batches = len(test_loader)
    final_cer = total_edit_err / num_batches
    final_em = total_exact_match / num_batches

    print("\n" + "=" * 45)
    print("                 FINAL TEST METRICS            ")
    print("=" * 45)
    print(f"Character Error Rate (CER):   {final_cer:.4%}")
    print(f"Exact Match (EM) Accuracy:    {final_em:.4%}")
    print("=" * 45)

    # =====================================================================
    # LIVE PILOT TESTING
    # =====================================================================
    # Grab a few qualitative predictions from the test set to display
    print("\nRunning live tests on individual images:")
    sample_images = [
        "data/crohme2019/test/ISICal19_1207_em_854.png",
        "data/crohme2019/test/ISICal19_1203_em_786.png",
        
    ]

    for sample_path in sample_images:
        path_to_check = os.path.join(DATA_DIR, sample_path) if not sample_path.startswith(DATA_DIR) else sample_path
        if os.path.exists(path_to_check):
            prediction = predict_latex(model, path_to_check, vocab, device)
            print(f"\nImage Path: {path_to_check}")
            print(f"Predicted LaTeX: {prediction}")
        else:
            print(f"\nSample visual file not found: {path_to_check}")


if __name__ == "__main__":
    main()