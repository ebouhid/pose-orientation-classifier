import os
import glob
import shutil
import argparse
import torch
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from orientation_model import OrientationModel  # Your model class


def load_and_preprocess_image(image_path, transform):
    """Load an image and apply preprocessing transforms."""
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    transformed = transform(image=image)
    return transformed["image"]


def evaluate_and_split_unstructured(model_path, input_dir, output_dir, batch_size=32):
    # Define positions and create corresponding output folders
    positions = ['back', 'front', 'left', 'right']
    for position in positions:
        os.makedirs(os.path.join(output_dir, position), exist_ok=True)

    # Albumentations transform for preprocessing
    test_transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ])

    # Load the model
    model = OrientationModel.load_from_checkpoint(model_path)
    model.eval()

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Gather all image paths from the input directory
    image_paths = glob.glob(os.path.join(input_dir, "*.*"))
    if not image_paths:
        print("No images found in the input directory.")
        return

    # Process images in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]

        # Load and preprocess the batch
        batch_images = [
            load_and_preprocess_image(image_path, test_transforms) for image_path in batch_paths
        ]
        batch_images = torch.stack(batch_images).to(device)

        # Model prediction
        with torch.no_grad():
            outputs = model(batch_images)
            outputs = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, dim=1).cpu().numpy()

        # Move files to the corresponding folders
        for filepath, pred in zip(batch_paths, predicted):
            position = positions[pred]
            dest_folder = os.path.join(output_dir, position)
            shutil.copy(filepath, dest_folder)  # Copy the file to the appropriate folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify and split unstructured images by orientation.")
    parser.add_argument("--input_dir", type=str, help="Input directory containing unstructured images.")
    parser.add_argument("--output_dir", type=str, help="Output directory to save classified images.")
    parser.add_argument("--model_path", type=str, help="Path to the trained model checkpoint.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing.")
    args = parser.parse_args()

    evaluate_and_split_unstructured(
        model_path=args.model_path,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
