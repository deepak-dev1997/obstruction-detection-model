import os
import shutil
import argparse
import cv2
import numpy as np
from glob import glob
from ultralytics import YOLO

def create_dataset(base_folder, mask_folder, dataset_dir,label_val=2, min_area=20):
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    image_files = glob(os.path.join(base_folder, "*.*"))
    num_annotated = 0

    for img_path in image_files:
        filename = os.path.basename(img_path)
        name, _ = os.path.splitext(filename)
        mask_path = os.path.join(mask_folder, filename)
        if not os.path.exists(mask_path):
            print(f"No mask found for {filename}, skipping.")
            continue

        img  = cv2.imread(img_path)                     # RGB image
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # keep channels
        if img is None or mask is None:
            print(f"Could not read {filename} or its mask.")
            continue

        # --- make sure the mask is single‑channel --------------------------------
        if len(mask.shape) == 3:            # (h, w, 3) or (h, w, 4)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # -------------------------------------------------------------------------

        h, w = mask.shape[:2]

        # pixels equal to the obstruction label become 1, everything else 0
        thresh = (mask == label_val).astype(np.uint8)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)

            boxes.append(
                f"0 {(x+bw/2)/w:.6f} {(y+bh/2)/h:.6f} {bw/w:.6f} {bh/h:.6f}"
            )

        if boxes:
            with open(os.path.join(labels_dir, f"{name}.txt"), "w") as f:
                f.write("\n".join(boxes))
            shutil.copy(img_path, os.path.join(images_dir, filename))
            num_annotated += 1
        else:
            print(f"No valid obstructions found in {filename}.")

    print(f"Dataset created with {num_annotated} annotated images.")
    return num_annotated 


def create_data_yaml(dataset_dir, output_yaml="data.yaml"):
    """
    Creates a data.yaml file for YOLO training using absolute paths.
    """
    abs_images_path = os.path.abspath(os.path.join(dataset_dir, "images"))
    data = {
        "train": abs_images_path,
        "val": abs_images_path,
        "nc": 1,
        "names": ["obstruction"]
    }
    with open(output_yaml, "w") as f:
        import yaml  # requires PyYAML (pip install pyyaml)
        yaml.dump(data, f)
    print(f"Created {output_yaml} for YOLO training with train/val paths: {abs_images_path}")

def main(base_folder, mask_folder, dataset_dir, epochs):
    parser = argparse.ArgumentParser(description="Train a YOLO model to detect rooftop obstructions.")
    
    parser.add_argument("--label_value", type=int, default=2,
                        help="Pixel value in mask that corresponds to the obstruction.")
    args = parser.parse_args()

    # Step 1: Create dataset structure and convert masks to YOLO annotations.
    num_annotated = create_dataset(base_folder, mask_folder,
                                   dataset_dir, args.label_value)

    # Step 2: Create a data.yaml file required by YOLO training.
    create_data_yaml(dataset_dir, output_yaml="data.yaml")

    # Step 3: Train the YOLO model using Ultralytics YOLO (using YOLOv8n pretrained weights).
    # The imgsz parameter is set to 250 since your images are 250x250.
    model = YOLO("yolov8n.pt")  # ensure you have the ultralytics package and model weights downloaded
    results = model.train(data="data.yaml", epochs=epochs, imgsz=512,device="cuda")

    # Corrected: use results.save_dir instead of results.run.dir to get the directory where weights are saved.
    best_model_path = os.path.join(results.save_dir, "weights", "best.pt")
    shutil.copy(best_model_path, "obstruction_yolo.pt")
    print(f"Training complete. {num_annotated} images with valid obstructions "
          f"were used. Model saved as obstruction_yolo.pt.")

if __name__ == "__main__":
    images = "trainimages"
    mask = "maskedimages"
    epoch = 50
    save_path = "saved_model"
    main(base_folder=images, mask_folder=mask, dataset_dir=save_path, epochs=epoch)
