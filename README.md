# Dogs vs Cats貓狗分類任務

專案中使用PyTorch建立模型，來分類圖片中的貓與狗

資料集來源：[Dogs vs Cats 資料集 (Kaggle)](https://www.kaggle.com/competitions/dogs-vs-cats/data)

---

### 1.先將專案clone下來：

```bash
git clone https://github.com/Larryyang727/dogs-vs-cats-work.git
```

### 2.安裝套件並開始訓練模型

```bash
cd dogs-vs-cats-work
```
```bash
pip install -r requirements.txt
```

#若未下載資料集，train.py會自動從Google Drive下載

```bash
python train.py --data_dir /content/data/dogs-vs-cats/
```
---
評估前若因訓練時間冗長，可以直接下載並使用已訓練好的模型檔案：
```bash
pip install gdown
gdown --id 1_qXVRdHmGWN0WfCcJCcfxRRZAgId5LFY
```

### 3.評估模型
```bash
python eval.py
```




## Model

- CrossEntropyLoss
- Adam optimizer (lr=1e-4)
- Early stopping with patience=3
- LR scheduler (ReduceLROnPlateau)

## Model Performance

| Model            | Accuracy | Precision | Recall |
|------------------|----------|-----------|--------|
| ResNet18         | 0.996    | 0.997     | 0.995  |
| EfficientNetB0   | 0.998    | 0.996     | 0.999  |
| **Ensemble**     | **0.999**| 0.998     | 0.999  |
