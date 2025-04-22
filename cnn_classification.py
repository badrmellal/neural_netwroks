import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Check if MPS (Metal Performance Shaders) is available on my mac
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS (Metal) device for acceleration on M1 Mac")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA device")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU device")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create directories for outputs
OUTPUT_DIR = "./outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "human_animal_classifier.pth")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32  # Smaller batch size for my M1 with 8GB RAM
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
NUM_WORKERS = 2  # Reduced for MacBook

# Define the classes we want to classify
CLASSES = ['man', 'woman', 'child', 'dog', 'cat', 'other']
NUM_CLASSES = len(CLASSES)

# Data Loading and Preprocessing

# Define transformations - optimized for efficiency
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Custom dataset class to handle multiple datasets from Huggingface
class HumanAnimalDataset(Dataset):
    def __init__(self, datasets, class_mapping, transform=None):
        """
        A dataset that combines multiple datasets and maps them to our target classes

        Args:
            datasets: List of loaded Huggingface datasets
            class_mapping: Dictionary mapping dataset-specific classes to our target classes
            transform: Image transformations to apply
        """
        self.datasets = datasets
        self.class_mapping = class_mapping
        self.transform = transform

        # Create a flat list of (dataset_idx, sample_idx, target_class) tuples
        self.samples = []
        for dataset_idx, dataset in enumerate(datasets):
            for sample_idx in range(len(dataset)):
                original_class = dataset[sample_idx]['label']
                # Map to our target classes if present in mapping
                if original_class in self.class_mapping[dataset_idx]:
                    target_class = self.class_mapping[dataset_idx][original_class]
                    self.samples.append((dataset_idx, sample_idx, target_class))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dataset_idx, sample_idx, target_class = self.samples[idx]
        sample = self.datasets[dataset_idx][sample_idx]

        # Get the image
        image = sample['image']

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, target_class


# Function to load datasets from Huggingface
def load_data_from_huggingface():
    print("Loading datasets from Huggingface...")

    # For people categories, we'll use a subset of the "beans" dataset just as a placeholder
    people_dataset = load_dataset("beans", split="train")

    # For animals, we'll use a subset of Oxford-IIIT Pet dataset
    animal_dataset = load_dataset("oxford-iiit-pet", split="train")

    # For "other" category, we can use a subset of CIFAR-100
    other_dataset = load_dataset("cifar100", split="train")

    print(f"Loaded datasets with sizes:")
    print(f"- People: {len(people_dataset)}")
    print(f"- Animals: {len(animal_dataset)}")
    print(f"- Others: {len(other_dataset)}")

    # Define class mappings from original datasets to our target classes
    # This is just a simplified mapping for demonstration

    # For people dataset (placeholder - using beans dataset)
    # In beans dataset: 0=angular_leaf_spot, 1=bean_rust, 2=healthy
    # We'll map these to our human categories just for demonstration
    people_mapping = {
        0: 0,  # Maps "angular_leaf_spot" to "man"
        1: 1,  # Maps "bean_rust" to "woman"
        2: 2,  # Maps "healthy" to "child"
    }

    # For animal dataset
    # In Oxford Pets, the labels are complex. We'll map a few breeds to dogs and cats
    # For simplicity, we'll say first 19 classes are cats and the rest are dogs
    animal_mapping = {}
    for i in range(37):
        if i < 19:
            animal_mapping[i] = 4  # Cat
        else:
            animal_mapping[i] = 3  # Dog

    # For other dataset (CIFAR-100), we'll map all classes to "other"
    other_mapping = {i: 5 for i in range(100)}  # All map to "other"

    # Combined mapping
    class_mapping = [people_mapping, animal_mapping, other_mapping]

    return [people_dataset, animal_dataset, other_dataset], class_mapping


# Create a balanced subset for faster training and testing
def create_balanced_subset(dataset, samples_per_class=100):
    class_samples = {cls: [] for cls in range(NUM_CLASSES)}

    # Collect indices for each class
    for idx in range(len(dataset)):
        _, cls = dataset[idx]
        if len(class_samples[cls]) < samples_per_class:
            class_samples[cls].append(idx)

    # Combine all indices
    subset_indices = []
    for cls in range(NUM_CLASSES):
        subset_indices.extend(class_samples[cls])

    return Subset(dataset, subset_indices)


# Main data loading function
def prepare_datasets():
    # Load datasets from Huggingface
    datasets, class_mapping = load_data_from_huggingface()

    # Create combined dataset
    full_dataset = HumanAnimalDataset(datasets, class_mapping, transform=val_transform)

    # Create a smaller balanced subset for training
    balanced_dataset = create_balanced_subset(full_dataset, samples_per_class=100)

    # Split into train and validation sets (80/20)
    train_size = int(0.8 * len(balanced_dataset))
    val_size = len(balanced_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        balanced_dataset, [train_size, val_size]
    )

    # Apply transformations
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, val_loader


# Model Definition

# Using MobileNetV3 - extremely efficient for devices like my M1 Mac
class HumanAnimalClassifier(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(HumanAnimalClassifier, self).__init__()

        # Load MobileNetV3 Small - very efficient because i have limited RAM
        if pretrained:
            self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        else:
            self.backbone = mobilenet_v3_small(weights=None)

        # Replace the classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# Training and Evaluation Functions
# ===============================

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_model(model, val_loader, criterion, device, class_names):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Save predictions and labels for classification report
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    # Generate classification report
    if len(all_preds) > 0:
        report = classification_report(
            all_labels, all_preds,
            target_names=class_names,
            zero_division=0
        )
        print("\nClassification Report:")
        print(report)

    return epoch_loss, epoch_acc, all_preds, all_labels


# Visualization Functions

def plot_metrics(history):
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_metrics.png'))
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()


def visualize_samples(data_loader, class_names, num_samples=8):
    # Get a batch of images
    images, labels = next(iter(data_loader))
    images = images[:num_samples]
    labels = labels[:num_samples]

    # Denormalize the images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)

    # Create a grid and show the images
    fig, axes = plt.subplots(2, num_samples // 2, figsize=(12, 6))
    axes = axes.flatten()

    for i, (img, label) in enumerate(zip(images, labels)):
        axes[i].imshow(img.permute(1, 2, 0).numpy())
        axes[i].set_title(class_names[label.item()])
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sample_images.png'))
    plt.close()


# Main Training Loop
# ================

def main():
    print("Starting Human-Animal Classification training...")

    # Prepare datasets and loaders
    train_loader, val_loader = prepare_datasets()
    print(f"Loaded {len(train_loader.dataset)} training samples and {len(val_loader.dataset)} validation samples")

    # Visualize some samples
    visualize_samples(train_loader, CLASSES)

    # Create model
    model = HumanAnimalClassifier(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    print(f"Created model: {model.__class__.__name__}")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, verbose=True
    )

    # Initialize history dict to track metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # Training loop
    best_val_loss = float('inf')

    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        # Train one epoch
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, DEVICE)

        # Validate
        val_loss, val_acc, val_preds, val_labels = validate_model(
            model, val_loader, criterion, DEVICE, CLASSES
        )

        # Update learning rate
        scheduler.step(val_loss)

        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved best model to {MODEL_PATH}")

    print("Training completed!")

    # Load best model for final evaluation
    model.load_state_dict(torch.load(MODEL_PATH))
    final_val_loss, final_val_acc, val_preds, val_labels = validate_model(
        model, val_loader, criterion, DEVICE, CLASSES
    )
    print(f"Final model - Val Loss: {final_val_loss:.4f}, Val Acc: {final_val_acc:.4f}")

    # Generate plots
    plot_metrics(history)
    plot_confusion_matrix(val_labels, val_preds, CLASSES)

    print(f"Saved results to {OUTPUT_DIR}")


# Inference Function for real-world usage
def predict_image(model, image_path, class_names, device):
    """Make a prediction on a single image"""
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    transform = val_transform
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, 1)

    predicted_class = class_names[prediction.item()]
    confidence = confidence.item()

    return predicted_class, confidence


if __name__ == "__main__":
    main()



# Usage:
# Load the trained model
model = HumanAnimalClassifier(num_classes=len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH))
model.to(DEVICE)

# Make predictions on new images
image_path = "/Users/macbook/Desktop/Projects/product-research.jpg"
predicted_class, confidence = predict_image(model, image_path, CLASSES, DEVICE)
print(f"Predicted: {predicted_class} with {confidence:.2%} confidence")
