import torch
import torch.optim as optim
from tqdm import trange
from models.model_def import get_resnet18, get_efficientnet_b0
from models.train_eval import train_one_epoch, validate, ensemble_evaluate
from utils.data_utils import get_dataloaders
import os
import zipfile
import gdown


if not os.path.exists("/content/data/dogs-vs-cats/train"):
    print("資料不存在，開始下載...")
    os.makedirs("/content/data", exist_ok=True)
    url = "https://drive.google.com/file/d/1fXwmnlPpqZKWVd8Tv8hw2ZHHqZQVhSbJ/view?usp=sharing"
    output = "/content/data/dogs-vs-cats.zip"
    gdown.download(url, output, fuzzy=True)
    # 解壓縮
    with zipfile.ZipFile("/content/data/dogs-vs-cats.zip", "r") as zip_ref:
        zip_ref.extractall("/content/data")
    print("資料下載並解壓縮完成！")
else:
    print("資料已存在，跳過下載。")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "/content/data/dogs-vs-cats/train"  #資料路徑
    batch_size = 32
    epochs = 30

    train_loader, val_loader, class_to_idx = get_dataloaders(data_dir, batch_size)
    print("類別對應：", class_to_idx)

    model_a = get_resnet18().to(device)
    model_b = get_efficientnet_b0().to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer_a = optim.Adam(model_a.parameters(), lr=1e-4)
    optimizer_b = optim.Adam(model_b.parameters(), lr=1e-4)

    scheduler_a = optim.lr_scheduler.ReduceLROnPlateau(optimizer_a, mode='min', patience=2, factor=0.5, verbose=True)
    scheduler_b = optim.lr_scheduler.ReduceLROnPlateau(optimizer_b, mode='min', patience=2, factor=0.5, verbose=True)

    early_stopping_patience = 3
    best_val_loss_a = float('inf')
    best_val_loss_b = float('inf')
    patience_counter_a = 0
    patience_counter_b = 0
    best_ensemble_acc = 0.0

    epoch_bar = trange(epochs, desc="Overall Training Progress")
    for epoch in epoch_bar:
        epoch_bar.set_description(f"Epoch {epoch+1}/{epochs}")

        train_loss_a = train_one_epoch(model_a, optimizer_a, criterion, train_loader, device)
        train_loss_b = train_one_epoch(model_b, optimizer_b, criterion, train_loader, device)

        val_loss_a, val_acc_a = validate(model_a, criterion, val_loader, device)
        val_loss_b, val_acc_b = validate(model_b, criterion, val_loader, device)
        ensemble_acc = ensemble_evaluate(model_a, model_b, val_loader, device)

        epoch_bar.set_postfix({
            "ResNet_Val": f"{val_acc_a:.4f}",
            "EffNet_Val": f"{val_acc_b:.4f}",
            "Ensemble": f"{ensemble_acc:.4f}"
        })

        print(f"\n  ResNet     - Train Loss: {train_loss_a:.4f}, Val Loss: {val_loss_a:.4f}, Val Acc: {val_acc_a:.4f}")
        print(f"  Efficient  - Train Loss: {train_loss_b:.4f}, Val Loss: {val_loss_b:.4f}, Val Acc: {val_acc_b:.4f}")
        print(f"  Ensemble   - Val Acc:    {ensemble_acc:.4f}")

        if ensemble_acc > best_ensemble_acc:
            best_ensemble_acc = ensemble_acc
            print(f"🎯 New best ensemble acc: {best_ensemble_acc:.4f} — model saved.")
            torch.save({
                'model_a_state_dict': model_a.state_dict(),
                'model_b_state_dict': model_b.state_dict(),
            }, 'best_ensemble.pth')

        scheduler_a.step(val_loss_a)
        scheduler_b.step(val_loss_b)

        if val_loss_a < best_val_loss_a:
            best_val_loss_a = val_loss_a
            patience_counter_a = 0
        else:
            patience_counter_a += 1

        if val_loss_b < best_val_loss_b:
            best_val_loss_b = val_loss_b
            patience_counter_b = 0
        else:
            patience_counter_b += 1

        if patience_counter_a >= early_stopping_patience and patience_counter_b >= early_stopping_patience:
            print("Early stopping triggered!")
            break

if __name__ == "__main__":
    main()
