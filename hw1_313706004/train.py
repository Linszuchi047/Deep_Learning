import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from model import ClassificationModel
import numpy as np


    
def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean = [0.4914, 0.4822, 0.4465] 
    std = [0.2470, 0.2435, 0.2616] 
    batch_size = 40
    n_epochs = 100

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize all images to 224x224
        # Random augmentations
        # Randomly rotate images by 40 degrees
        transforms.RandomRotation(40),
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),  # Random color jitter
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=mean, std=std)  # Normalize with mean and std
    ])

    train_dir = r'C:\Users\User\DeepLearning\Deep_Learning\HW1\train' 
    all_train = datasets.ImageFolder(root=train_dir, transform = train_transform)
    # test = datasets.ImageFolder(root = 'ttest', transform = train_transform)
    train_size = int(0.9 * len(all_train))
    validation_size = len(all_train) - train_size
    train_dataset, validation_dataset = random_split(all_train , [train_size, validation_size])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=3
    )
    val_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=3
    )
    model = ClassificationModel().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    # Early stopping class
    class EarlyStopper:
        def __init__(self, patience=7, min_delta=0):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.min_validation_loss = np.inf

        def early_stop(self, validation_loss):
            if validation_loss < self.min_validation_loss:
                self.min_validation_loss = validation_loss
                self.counter = 0
            elif validation_loss > (self.min_validation_loss + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            return False

    # Train function
    def training(model, train_loader, optimizer, loss_fn):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        accuracy = correct / total
        return epoch_loss, accuracy

    
    @torch.no_grad()
    def validate(model, val_loader, loss_fn):
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item() * images.size(0)
            
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
        avg_loss = val_loss / len(val_loader.dataset)
        accuracy = correct / total
        return avg_loss, accuracy

    # Training loop
    train_loss_list = []
    valid_loss_list = []
    early_stopper = EarlyStopper(patience=7)

    best_val_acc = 0.0

    for epoch in range(n_epochs):
        train_loss, train_acc = training(model, train_loader, optimizer, loss_fn)
        val_loss, val_acc = validate(model, val_loader, loss_fn)
        
        train_loss_list.append(train_loss)
        valid_loss_list.append(val_loss)
        
        print(f"Epoch {epoch+1}/{n_epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Step the learning rate scheduler
        scheduler.step()
        
        # Early stopping check
        if early_stopper.early_stop(val_loss):
            print("Early stopping")
            break
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "w_313706004.pth")

    # Save the final model
    torch.save(model.state_dict(), "wf_313706004_model.pth")

if __name__ == "__main__":
    train()
