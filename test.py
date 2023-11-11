import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torchmetrics import Accuracy
from dataset import CustomDataset
from train import ResNet18
from omegaconf import DictConfig
import hydra

def load_class_names(train_dataset_path):
    # Load class names from the training dataset
    train_dataset = CustomDataset(root=train_dataset_path)
    class_names = train_dataset.classes
    return class_names

def preprocess_image(image_path, transform=None):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return input_tensor

def predict_class(model, input_tensor):
    # Make predictions
    with torch.no_grad():
        input_tensor = input_tensor.to(model.device)
        model_output = model(input_tensor)
        predicted_class = torch.argmax(model_output, dim=1).item()

    return predicted_class

@hydra.main(config_path="configs",config_name="train")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Specify the path to the saved model
    model_path = '/home/saeed101/projects/ml/outputs/2023-11-11/12-43-30/model.pth'
    
    train_dataset_path = '/home/saeed101/projects/ml/dataset/train'

    # Specify the path to the test dataset
    image_path = "/home/saeed101/projects/ml/dataset/test/german shepherd/gs44_jpg.rf.fdcf10494ba04557bb52ed54f036e794.jpg"
    # Load class names
    class_names = load_class_names(train_dataset_path)

    # Transformations for the test dataset
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


    model = ResNet18(num_classes=cfg.model.num_classes, optimizer_cfg=cfg.model.optimizer)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    

    input_tensor = preprocess_image(image_path=image_path, transform= test_transform)
    

    # Make predictions
    predicted_class_idx = predict_class(model, input_tensor)
    
    # Display the predicted class name
    predicted_class_name = class_names[predicted_class_idx]
    print(f"The predicted class is: {predicted_class_name}")

if __name__ == "__main__":
    main()
