import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_ensemble(model_a, model_b, val_loader, device):
    model_a.eval()
    model_b.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            out1 = model_a(images)
            out2 = model_b(images)
            avg_out = (out1 + out2) / 2
            probs = torch.softmax(avg_out, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # 取 class 1 的機率

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def plot_confusion_matrix(y_true, y_pred, classes=['Cat', 'Dog'], save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path)
    plt.show()

def plot_roc_curve(y_true, y_probs, save_path='roc_curve.png'):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    import torch
    from models.load_model import load_models
    from utils.data_utils import get_val_loader  # 假設你有這個函式幫你載入資料
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_a, model_b = load_models("best_ensemble.pth", device)
    val_loader = get_val_loader()  # 自訂義你的資料載入函式

    y_true, y_pred, y_probs = evaluate_ensemble(model_a, model_b, val_loader, device)

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))

    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_probs)

