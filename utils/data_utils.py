from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
#套用IMAGENET訓練資料集的標準差和平均值
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), #大多模型預設輸入224
        transforms.RandomHorizontalFlip(), #隨機翻轉
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_transform, val_transform

def get_dataloaders(data_dir, batch_size=32, val_ratio=0.2):
    train_transform, val_transform = get_transforms()
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    
    train_len = int((1 - val_ratio) * len(full_dataset))
    val_len = len(full_dataset) - train_len
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])
    
    #套用不同transform
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, full_dataset.class_to_idx

