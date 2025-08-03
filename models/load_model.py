import torch
from .model_def import get_resnet18, get_efficientnet_b0

def load_models(weight_path, device):
    # 建立模型架構（不載入預訓練權重，改用 False 方便載入自訓練權重）
    model_a = get_resnet18(pretrained=False)
    model_b = get_efficientnet_b0(pretrained=False)

    # 載入訓練好的權重
    checkpoint = torch.load(weight_path, map_location=device)
    model_a.load_state_dict(checkpoint['model_a_state_dict'])
    model_b.load_state_dict(checkpoint['model_b_state_dict'])

    # 放到裝置上並設為評估模式
    model_a = model_a.to(device)
    model_b = model_b.to(device)
    model_a.eval()
    model_b.eval()

    return model_a, model_b
