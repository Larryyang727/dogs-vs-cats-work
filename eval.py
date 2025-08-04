import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
import argparse

def evaluate_model(model, val_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
#儲存GT標籤、預測與機率
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1) #預測機率
            preds = torch.argmax(probs, dim=1) #類別預測

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  #是狗的機率 ROC

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)
#兩個模型做平均
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
            all_probs.extend(probs[:, 1].cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def print_metrics(name, y_true, y_pred):
    print(f"\n{name} Performance:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))

def plot_confusion_matrix(y_true, y_pred, classes=['Cat','Dog'], title='Confusion Matrix', save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_roc_curve(y_true, y_probs, label='ROC', save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/content/data/dogs-vs-cats/train')
    parser.add_argument('--model_path', type=str, default='best_ensemble.pth')
    args = parser.parse_args()
    from models.load_model import load_models
    from utils.data_utils import get_dataloaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_a, model_b = load_models(
      "best_ensemble.pth", device)  #ResNet18、EfficientNetB0

    train_dir = args.data_dir
    train_path = train_dir

    _, val_loader, _ = get_dataloaders(train_path,batch_size=32,val_ratio=0.2)

    #ResNet18
    y_true_a, y_pred_a, y_probs_a = evaluate_model(model_a, val_loader, device)
    print_metrics("ResNet18", y_true_a, y_pred_a)
    plot_confusion_matrix(y_true_a, y_pred_a, title='ResNet18 Confusion Matrix')
    plot_roc_curve(y_true_a, y_probs_a, label='ResNet18')

    #EfficientNetB0
    y_true_b, y_pred_b, y_probs_b = evaluate_model(model_b, val_loader, device)
    print_metrics("EfficientNetB0", y_true_b, y_pred_b)
    plot_confusion_matrix(y_true_b, y_pred_b, title='EfficientNetB0 Confusion Matrix')
    plot_roc_curve(y_true_b, y_probs_b, label='EfficientNetB0')

    #Ensemble
    y_true_ens, y_pred_ens, y_probs_ens = evaluate_ensemble(model_a, model_b, val_loader, device)
    print_metrics("Ensemble", y_true_ens, y_pred_ens)
    plot_confusion_matrix(y_true_ens, y_pred_ens, title='Ensemble Confusion Matrix')
    plot_roc_curve(y_true_ens, y_probs_ens, label='Ensemble')

