import torch
from model import ClassificationModel
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import os
import pandas as pd


def test():
    # Load the trained model
    model = ClassificationModel()  # Initialize the model with 100 classes
    model.load_state_dict(torch.load('w_313706004.pth'))  # Assuming you saved the trained model as model.pth
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    mean = [0.4914, 0.4822, 0.4465] 
    std = [0.2470, 0.2435, 0.2616] 

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
    path = r"C:\Users\User\DeepLearning\Deep_Learning\HW1\ttest"
    test = datasets.ImageFolder(root = path, transform = train_transform)
    test_loader = DataLoader(
    test,
    batch_size=40,
    shuffle=True,
    num_workers=3
)

    # # Data transformations
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])

    # Load the test dataset
    # test_dataset = datasets.ImageFolder(root='ttest', transform=transform)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    total, correct_top5 = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # _, predicted = torch.max(outputs, 1)
            _, predicted_top5 = torch.topk(outputs, 5, dim=1)
            for i in range(labels.size(0)):
                if labels[i] in predicted_top5[i]:
                    correct_top5 += 1
            total += labels.size(0)
            # correct += (predicted == labels).sum().item()

    # accuracy = correct / total
    # print(f'Test Accuracy: {accuracy * 100:.2f}%')
    top5_accuracy = correct_top5 / total
    print(f'Test Top-5 Accuracy: {top5_accuracy * 100:.2f}%')

    class TestDataset(Dataset):
        def __init__(self, test_dir, transform=None):
            self.test_dir = test_dir
            self.file_names = os.listdir(test_dir)
            self.transform = transform

        def __len__(self):
            return len(self.file_names)

        def __getitem__(self, idx):
            img_name = self.file_names[idx]
            img_path = os.path.join(self.test_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, img_name

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClassificationModel() 
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    model = model.to(device)

    # Define transformations for train and test datasets
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize all images to 224x224
        # Random augmentations
        # Randomly rotate images by 40 degrees
        transforms.RandomRotation(40),
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),  # Random color jitter
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=mean, std=std)  # Normalize with mean and std
    ])

    # Load the train dataset using ImageFolder
    train_dir = os.path.abspath('./train')  # Replace with the path to your train dataset
    train_dataset = ImageFolder(train_dir, transform=transform)

    # Get the class_to_idx from the training dataset and invert it to create idx_to_class
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}  # Invert class_to_idx to idx_to_class

    # Load test dataset
    test_dir = os.path.abspath('./test')  # Path to your test dataset
    test_dataset = TestDataset(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Prepare CSV for results
    results = []

    # Make predictions and store them in result.csv
    with torch.no_grad():
        for images, file_names in test_loader:  # file_names returned from the dataset
            images = images.to(device)  # Move images to GPU/CPU
            outputs = model(images)
            
            # Get top-5 predictions for each image
            _, top5_preds = torch.topk(outputs, 5, dim=1)
            
            for i, preds in enumerate(top5_preds):
                file_name = file_names[i]  # Get the file name
                # Convert predicted indices to class labels (words)
                pred_labels = [idx_to_class[pred.item()] for pred in preds]
                
                # Append to results
                results.append([file_name] + pred_labels)

    # Save results to CSV
    df = pd.DataFrame(results, columns=['file_name', 'pred1', 'pred2', 'pred3', 'pred4', 'pred5'])
    df.to_csv('result.csv', index=False)

    print("Predictions saved to result.csv")


if __name__ == "__main__":
    test()
