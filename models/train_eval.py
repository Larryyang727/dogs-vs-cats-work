import torch
from tqdm import trange

def train_one_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Train loss: {epoch_loss:.4f}")
    return epoch_loss

def validate(model, criterion, val_loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_loss = running_loss / len(val_loader.dataset)
    val_acc = correct / total
    print(f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")
    return val_loss, val_acc

def ensemble_evaluate(model1, model2, val_loader, device):
    model1.eval()
    model2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            out1 = model1(images)
            out2 = model2(images)
            avg_out = (out1 + out2) / 2
            _, preds = torch.max(avg_out, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total
