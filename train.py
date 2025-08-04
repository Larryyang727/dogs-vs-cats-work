import torch
import torch.optim as optim
from tqdm import trange
from models.model_def import get_resnet18, get_efficientnet_b0
from models.train_eval import train_one_epoch, validate, ensemble_evaluate
from utils.data_utils import get_dataloaders
import os
import zipfile
import gdown
import argparse
import shutil

def download_data_if_needed(data_root):
    train_path = os.path.join(data_root, "train")
    if not os.path.exists(train_path):
        os.makedirs(data_root, exist_ok=True)
        url = "https://drive.google.com/uc?id=1fXwmnlPpqZKWVd8Tv8hw2ZHHqZQVhSbJ"
        output = os.path.join(data_root, "dogs-vs-cats.zip")
        gdown.download(url, output, fuzzy=True)

        with zipfile.ZipFile(output, "r") as zip_ref:
            zip_ref.extractall(data_root)

        extracted_dir = os.path.join(data_root,"dogs-vs-cats")
        if os.path.exists(extracted_dir):
            for item in os.listdir(extracted_dir):
                shutil.move(os.path.join(extracted_dir, item), data_root)
            shutil.rmtree(extracted_dir)
        print("下載並解壓完成！")


def main(data_dir, epochs=30, batch_size=32):
    download_data_if_needed(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_path = os.path.join(data_dir, "train")
    train_loader, val_loader, class_to_idx = get_dataloaders(train_path, batch_size)
    print("類別對應：", class_to_idx) #顯示資料夾名稱對應的類別編號

    model_a = get_resnet18().to(device)
    model_b = get_efficientnet_b0().to(device)

    criterion = torch.nn.CrossEntropyLoss() #用CrossEntropyLoss當損失函數
    optimizer_a = optim.Adam(model_a.parameters(), lr=1e-4)#ADAM優化器
    optimizer_b = optim.Adam(model_b.parameters(), lr=1e-4)

    scheduler_a = optim.lr_scheduler.ReduceLROnPlateau(optimizer_a, mode='min', patience=2, factor=0.5, verbose=True)#觀察LOSS自動調整LR，2次沒進步就調整
    scheduler_b = optim.lr_scheduler.ReduceLROnPlateau(optimizer_b, mode='min', patience=2, factor=0.5, verbose=True)

    early_stopping_patience = 3 #3次EPOCH沒進步或降低停止訓練
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
#新的ensemble表現更好就儲存新的
        if ensemble_acc > best_ensemble_acc:
            best_ensemble_acc = ensemble_acc
            print(f"New best ensemble acc: {best_ensemble_acc:.4f} — model saved.")
            torch.save({
                'model_a_state_dict': model_a.state_dict(),
                'model_b_state_dict': model_b.state_dict(),
            }, 'best_ensemble.pth')

        scheduler_a.step(val_loss_a)
        scheduler_b.step(val_loss_b)
#改善就把patience歸0，沒改善+1
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
    import argparse
    parser = argparse.ArgumentParser(description="Dogs vs Cats Training")
    parser.add_argument("--data_dir", type=str, default="/content/data/dogs-vs-cats", help="資料集根目錄路徑")
    parser.add_argument("--epochs", type=int, default=30, help="訓練輪數")
    parser.add_argument("--batch_size", type=int, default=32, help="batch大小")
    args = parser.parse_args()


    main(args.data_dir, args.epochs, args.batch_size)
