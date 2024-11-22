import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import OrientationDataset  # Your custom dataset class
from orientation_model import OrientationModel  # Your model class

def evaluate_model(model_path, assets_dir, split_ids, position, batch_size=32, num_workers=4):
    test_transforms = A.Compose([
    A.Resize(224, 224),  # Resize to match model input
    A.Normalize(), # Normalize the image (same as training)
    ToTensorV2()  # Convert to PyTorch tensor
])

    test_dataset = OrientationDataset(
        assets_dir=assets_dir,
        split_csv=split_ids,
        transform=test_transforms,
        verbose=False
    )

    # Filter dataset by the specific position
    test_dataset.data = [data for data in test_dataset.data if data[1] == position]

    # Create DataLoader for the test dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Load the model
    model = OrientationModel.load_from_checkpoint(model_path)
    model.eval()

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    correct = 0
    total = 0

    # Evaluate accuracy
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            labels = torch.argmax(labels, dim=1)

            outputs = model(images)
            outputs = torch.softmax(outputs, dim=1)
            # print("outputs:", outputs)
            predicted = torch.argmax(outputs, dim=1)
            # print("predicted:", predicted)
            # print("labels:", labels)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Accuracy for position {position}: {accuracy * 100:.2f}%')

    return accuracy

if __name__ == "__main__":
    model_path = "lightning_logs/version_11/checkpoints/epoch=3-step=196.ckpt"  # Path to your trained model
    assets_dir = "./merged_real_clinics_assets"  # Directory containing asset folders
    split_ids = "split_csvs/test.csv"  # Path to the test split CSV file
    
    for position in ['back', 'front', 'left', 'right']:
        evaluate_model(model_path, assets_dir, split_ids, position)
