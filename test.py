import torch
from torchvision import transforms
from PIL import Image
import os
import pandas as pd

#模型
from models.load_model import load_models  # 假設你有這個function

def predict_ensemble_test(model_a, model_b, test_dir, device, output_file='submission.csv'):
    model_a.eval()
    model_b.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image_files = sorted(os.listdir(test_dir), key=lambda x: int(x.split('.')[0]))
    results = []

    with torch.no_grad():
        for fname in image_files:
            img_path = os.path.join(test_dir, fname)
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)

            out_a = model_a(image)
            out_b = model_b(image)
            probs = (torch.softmax(out_a, dim=1) + torch.softmax(out_b, dim=1)) / 2
            pred = torch.argmax(probs, dim=1).item()

            img_id = int(fname.split('.')[0])
            results.append([img_id, pred])

    df = pd.DataFrame(results, columns=['id', 'label'])
    df.to_csv(output_file, index=False)
    print(f'Submission saved {output_file}')


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_a, model_b = load_models("best_ensemble.pth", device)  #訓練好的模型
    test_dir = '/content/data/dogs-vs-cats/test1'  #測試資料夾路徑
    predict_ensemble_test(model_a, model_b, test_dir, device)
