import os
import time
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Folder Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # aps360-hmer
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "data"))  # aps360-hmer/data
CROHME_DIR = os.path.join(DATA_DIR, "crohme2019")  # aps360-hmer/data/crohme2019

# Ground Truth Paths
TXT_PATHS = {
    "train": os.path.join(CROHME_DIR, "crohme2019_train.txt"),
    "val": os.path.join(CROHME_DIR, "crohme2019_valid.txt"),
    "test": os.path.join(CROHME_DIR, "crohme2019_test.txt"),
}

# =====================================================================
# 1. ROBUST METRIC COMPUTATION
# =====================================================================
def levenshtein_dist(s1, s2):
    """Computes token-based insertion, deletion, and substitution edit distance."""
    tokens1 = s1.split()
    tokens2 = s2.split()

    if len(tokens1) < len(tokens2):
        return levenshtein_dist(s2, s1)
    if len(tokens2) == 0:
        return len(tokens1)

    previous_row = range(len(tokens2) + 1)
    for i, t1 in enumerate(tokens1):
        current_row = [i + 1]
        for j, t2 in enumerate(tokens2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (t1 != t2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_metrics(predictions, ground_truths):
    """Calculates evaluation metrics (Exact Match, Levenshtein, and CER)."""
    exact_matches = 0
    total_levenshtein = 0
    total_ref_tokens = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_stripped = pred.strip()
        gt_stripped = gt.strip()

        if pred_stripped == gt_stripped:
            exact_matches += 1

        dist = levenshtein_dist(pred_stripped, gt_stripped)
        total_levenshtein += dist
        total_ref_tokens += max(len(gt_stripped.split()), 1)

    exact_match_acc = (exact_matches / len(ground_truths)) * 100
    mean_levenshtein = total_levenshtein / len(ground_truths)
    cer = (total_levenshtein / total_ref_tokens) * 100

    return {
        "exact_match_pct": exact_match_acc,
        "mean_levenshtein": mean_levenshtein,
        "cer_pct": cer,
    }


# =====================================================================
# 2. FILE PARSING & DATA SPLIT LOADING
# =====================================================================
def load_crohme_split(txt_file_path):
    """Parses tab-separated text dataset tracking metadata."""
    image_paths = []
    target_sequences = []

    if not os.path.exists(txt_file_path):
        raise FileNotFoundError(
            f"Could not find metadata file at: {txt_file_path}"
        )

    with open(txt_file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t", 1)
            if len(parts) != 2:
                print(
                    f"Skipping malformed row {line_num} in {os.path.basename(txt_file_path)}"
                )
                continue

            rel_inkml_path, target_latex_expr = parts

            # "crohme2019/test/XYZ.inkml" -> "crohme2019/test/XYZ.png"
            rel_png_path = rel_inkml_path.replace(".inkml", ".png")

            # Join with parent `data/` folder directory
            full_png_path = os.path.normpath(
                os.path.join(DATA_DIR, rel_png_path)
            )

            image_paths.append(full_png_path)
            target_sequences.append(target_latex_expr)

    return image_paths, target_sequences


# =====================================================================
# 3. GLOBAL MODEL RETRIEVAL CLASS
# =====================================================================
class GlobalKNNBaseline:

    def __init__(self, k=1, target_shape=(32, 100)):
        self.k = k
        self.target_shape = target_shape
        # 'euclidean' metrics are immune to Zero Division on empty/all-black arrays
        self.model = KNeighborsClassifier(n_neighbors=k, metric="euclidean")

    def preprocess_image(self, img_path):
        """Loads physical file, downsamples, and returns a flattened 1D array."""
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image vector source missing: {img_path}")

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, self.target_shape)
        img_normalized = img_resized.astype(np.float32) / 255.0
        return img_normalized.flatten()

    def fit(self, X_train_paths, y_train_labels):
        print("Preprocessing training frames into arrays...")
        X_train_features = []
        valid_y_labels = []

        for p, label in zip(X_train_paths, y_train_labels):
            try:
                feat = self.preprocess_image(p)
                X_train_features.append(feat)
                valid_y_labels.append(label)
            except FileNotFoundError:
                continue

        X_train_features = np.array(X_train_features)
        y_train_labels = np.array(valid_y_labels)

        print(
            f"Fitting KNN model on {len(X_train_features)} verified samples..."
        )
        self.model.fit(X_train_features, y_train_labels)

    def predict_split(self, X_paths):
        predictions = []
        for path in X_paths:
            try:
                feat = self.preprocess_image(path).reshape(1, -1)
                pred = self.model.predict(feat)[0]
                predictions.append(pred)
            except Exception:
                # Append empty response to preserve array alignment on corrupt images
                predictions.append("")
        return predictions


# =====================================================================
# 4. RUNNER PIPELINE
# =====================================================================
if __name__ == "__main__":
    print("=== STEP 1: PARSING TXT DATA FILES ===")
    try:
        train_paths, train_gt = load_crohme_split(TXT_PATHS["train"])
        val_paths, val_gt = load_crohme_split(TXT_PATHS["val"])
        test_paths, test_gt = load_crohme_split(TXT_PATHS["test"])
    except FileNotFoundError as e:
        print(f"\n[FATAL ERROR] {e}")
        print("Please check that your text files exist in data/crohme2019/")
        exit(1)

    print(f"Detected Train Examples: {len(train_paths)}")
    print(f"Detected Val Examples:   {len(val_paths)}")
    print(f"Detected Test Examples:  {len(test_paths)}")

    # Handshake Diagnostic Check
    print("\n--- Running path mapping test check ---")
    if len(train_paths) > 0:
        first_img = train_paths[0]
        print(f"Primary image path target: {first_img}")
        if os.path.exists(first_img):
            print("Mapping status: SUCCESS! Images found and linked correctly.")
        else:
            print("Mapping status: FAILED.")
            print(
                f"No image found at: {first_img}\n"
                "Please verify that your processed PNGs are inside the "
                "'data/crohme2019/train' directory."
            )
            exit(1)

    # Initialize model with Euclidean metric
    # baseline = GlobalKNNBaseline(k=1, target_shape=(32, 100))
    baseline = GlobalKNNBaseline(k=1, target_shape=(100, 32))

    print("\n=== STEP 2: TRAINING (INDEXING TRAINING SPLIT) ===")
    start_time = time.time()
    baseline.fit(train_paths, train_gt)
    print(f"Indexing completed in {time.time() - start_time:.2f}s")

    # Evaluate datasets
    splits_to_test = {
        "TRAIN DATASET (MEMORIZATION CHECK)": (train_paths, train_gt),
        "VALIDATION DATASET": (val_paths, val_gt),
        "TEST DATASET": (test_paths, test_gt),
    }

    print("\n=== STEP 3: RUNNING EVALUATION ===")
    for split_name, (paths, ground_truths) in splits_to_test.items():
        if len(paths) == 0:
            print(f"Skipping {split_name} (no files loaded).")
            continue

        print(f"\nEvaluating performance on {split_name}...")
        start_eval = time.time()

        preds = baseline.predict_split(paths)
        metrics = calculate_metrics(preds, ground_truths)

        print(f"Completed in {time.time() - start_eval:.2f} seconds")
        print(f"  - Exact Match (EM): {metrics['exact_match_pct']:.2f}%")
        print(
            f"  - Mean Levenshtein Distance: {metrics['mean_levenshtein']:.2f} tokens"
        )
        print(f"  - Character Error Rate (CER): {metrics['cer_pct']:.2f}%")