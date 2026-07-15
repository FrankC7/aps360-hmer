import csv
import glob
import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np


def parse_inkml(file_path):
    """Parses InkML to extract handwritten strokes and the LaTeX ground truth."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML for {file_path}: {e}")
        return None, None

    ns = {"ink": "http://www.w3.org/2003/InkML"}

    strokes = []
    for trace in root.findall(".//ink:trace", ns):
        text = trace.text.strip()
        coords = []
        for point in text.replace("\n", ",").split(","):
            point = point.strip()
            if not point:
                continue
            parts = point.split()
            if len(parts) >= 2:
                try:
                    coords.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
        if coords:
            strokes.append(coords)

    latex_truth = ""
    for annot in root.findall(".//ink:annotation", ns):
        if annot.attrib.get("type") == "truth":
            latex_truth = annot.text.strip()
            break

    return strokes, latex_truth


def render_to_canvas(strokes, target_h=128, target_w=400, stroke_thickness=2):
    """Renders coordinates into an aspect-preserved space."""
    all_coords = [pt for stroke in strokes for pt in stroke]
    if not all_coords:
        return np.zeros((target_h, target_w), dtype=np.uint8)

    all_coords = np.array(all_coords)
    min_x, min_y = np.min(all_coords, axis=0)
    max_x, max_y = np.max(all_coords, axis=0)

    width = max(int(max_x - min_x), 1)
    height = max(int(max_y - min_y), 1)

    margin = 10
    temp_img = np.zeros(
        (height + 2 * margin, width + 2 * margin), dtype=np.uint8
    )

    for stroke in strokes:
        points = [
            (int(x - min_x + margin), int(y - min_y + margin))
            for x, y in stroke
        ]
        for i in range(len(points) - 1):
            cv2.line(
                temp_img,
                points[i],
                points[i + 1],
                color=255,
                thickness=stroke_thickness,
                lineType=cv2.LINE_AA,
            )

    h, w = temp_img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(int(w * scale), 1)
    new_h = max(int(h * scale), 1)

    resized = cv2.resize(temp_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    start_y = (target_h - new_h) // 2
    start_x = (target_w - new_w) // 2
    canvas[start_y : start_y + new_h, start_x : start_x + new_w] = resized

    return canvas


def preserved_batch_convert(input_dir, output_dir, image_format="png"):
    """Finds all inkml files, converts them, and preserves subfolder structures."""
    # Find all .inkml files recursively
    query = os.path.join(input_dir, "**", "*.inkml")
    inkml_files = glob.glob(query, recursive=True)

    if not inkml_files:
        print(f"No .inkml files found in '{input_dir}'!")
        return

    print(f"Found {len(inkml_files)} files. Starting split-aware conversion...")

    # Dictionary to hold independent lists of metadata per folder split
    # e.g., {'train': [], 'test': [], 'valid': []}
    split_metadata = {}

    for i, file_path in enumerate(inkml_files):
        # Determine relative folder tree from input_dir
        # If file_path is 'archive/.../train/math.inkml', rel_dir will be 'train'
        rel_path = os.path.relpath(file_path, input_dir)
        rel_dir = os.path.dirname(rel_path)

        # Build clean output subfolder path
        target_subfolder = os.path.join(output_dir, rel_dir)
        os.makedirs(target_subfolder, exist_ok=True)

        # File naming
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        img_name = f"{base_name}.{image_format}"
        img_path = os.path.join(target_subfolder, img_name)

        # Convert
        strokes, latex = parse_inkml(file_path)
        if not strokes:
            continue

        image_data = render_to_canvas(strokes)
        cv2.imwrite(img_path, image_data)

        # Track metadata separately by split-key (e.g., 'train', 'test', 'valid')
        # We find the top-level directory name under input_dir to use as the split key
        split_key = rel_dir.split(os.sep)[0] if rel_dir else "default"

        if split_key not in split_metadata:
            split_metadata[split_key] = []

        split_metadata[split_key].append({"file_name": img_name, "latex": latex})

        if (i + 1) % 100 == 0 or (i + 1) == len(inkml_files):
            print(f"Processed {i+1}/{len(inkml_files)} files...")

    # Save matching metadata.csv files inside each target partition folder
    for split, records in split_metadata.items():
        split_csv_path = os.path.join(output_dir, split, "metadata.csv")
        with open(split_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "latex"])
            writer.writeheader()
            writer.writerows(records)
        print(f"-> Generated: {split_csv_path} ({len(records)} entries)")

    print("\nAll done! Subfolder structures successfully preserved.")


if __name__ == "__main__":
    # Point this to your top level directory matching your image
    SOURCE_FOLDER = "./archive/crohme2019/crohme2019"
    DESTINATION_FOLDER = "./crohme_processed"

    preserved_batch_convert(
        input_dir=SOURCE_FOLDER,
        output_dir=DESTINATION_FOLDER,
        image_format="png",
    )