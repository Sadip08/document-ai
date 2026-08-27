import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

def train_doc_classifier(data_dir, num_epochs=10):
    # Define data transformations
    # ResNet expects 224x224 images, so we resize and normalize the images accordingly
    data_transforms = {
        'train' :  transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    print(f"Loading training data from {data_dir}")
    image_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
    dataloader = DataLoader(image_dataset, batch_size=32, shuffle=True, num_workers=4)

    dataset_size = len(image_dataset)
    class_names = image_dataset.classes
    print(f"Found {dataset_size} images belonging to {len(class_names)} classes: {class_names}")

    # Load a pre-trained ResNet18
    print("Loading pre-trained ResNet model")
    model = models.resnet18(weights= models.ResNet18_Weights.DEFAULT)

    # Modify the final layer to match the number of classes in our dataset
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(class_names))

    # Move the model to GPU if available
    device = torch.device("cuda")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.double() / dataset_size

        print(f'Epoch {epoch}/{num_epochs - 1} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}')

    save_path = os.path.join('..', '..', 'models', 'doc_classifier_resnet18.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names
    }, save_path)

    print(f"Training complete! Model saved to {save_path}")
    return model, class_names

if __name__ == "__main__":
    data_dir = "data/splits/doc_classification"
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")
    train_doc_classifier(data_dir, num_epochs=10)