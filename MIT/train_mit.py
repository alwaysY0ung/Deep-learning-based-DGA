import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
import pytz

korean_tz = pytz.timezone('Asia/Seoul')
now_korea = datetime.datetime.now(korean_tz)
timestamp = now_korea.strftime('%Y%m%d_%H%M%S') # YYYYMMDD_HHMMSS 형식으로 타임스탬프 생성
timestamp_run = timestamp[4:13] # MMDD_HHMM 형식으로

# hyperparameter 1
percentile_value = '0.999'


# wandb
import wandb
wandb.init(
    project="dga-detection-mit",
    name=f"MIT_{timestamp_run}",
    config={
        "learning_rate": 1e-3,
        "epochs": 15,
        "batch_size": 100,
        "model_architecture": "MIT",
        "benign": "alexa_top_1m",
        "dga": "harpomaxx",
        "train_size": 0.8,
        "validation_size": 0.1,
        "test_size": 0.1,
        "threshold_type": f"{percentile_value*100}th_percentile_FPR",
        "loss_function": "BCELoss",
        "optimizer": "Adam",

    }
)

# load all dataset
df_benign = pd.read_parquet('./cache/alexa.parquet') # originally 1000000 rows since alexa_top_1m_170320.csv has 1 million rows
df_dga = (pd.read_parquet('./cache/harpomaxx.parquet'))
df_dga = df_dga.sample(n=1000000, random_state=42) # originally 1915335 rows, so just sample 1 million rows for balance
df = pd.concat([df_benign, df_dga]).reset_index(drop=True) # axis=0(행 방향으로 합치는 것)이 default라 적지 않아도 됨.

print(df.head(20))

df["domain"] = df["domain"].str.lower()
df["domain"] = df["domain"].str.zfill(75) # 패딩: add zero padding to the left # cf. there's no domain names over 75 characters
df["domain"] = df["domain"].apply(lambda x: [ord(c) for c in x]) # convert each character to its ASCII integer value using ord() # python list

print(df.head(20))

# gonna use train, val, test in a 8:1:1 ratio (so 10 thousand for test)
train, temp = train_test_split(
    df, test_size=0.2, 
    random_state=42, 
    stratify=df['label']) # stratify=df['label'] : train과 test에 label 비율이 동일하게 분포되도록 나눔

val, test = train_test_split(
    temp, test_size=0.5,
    random_state=42,
    stratify=temp['label'] # 마찬가지로 1:1 비율이 되도록 stratify 적용
)

# 각 데이터셋의 크기 확인
print(f"훈련 데이터셋 크기: {len(train)}")
print(f"검증 데이터셋 크기: {len(val)}")
print(f"테스트 데이터셋 크기: {len(test)}")

# 각 데이터셋의 benign/dga 비율 확인
print("\n--- 데이터셋별 비율 확인 ---")
print("훈련 데이터셋 비율:\n", train['label'].value_counts(normalize=True))
print("검증 데이터셋 비율:\n", val['label'].value_counts(normalize=True))
print("테스트 데이터셋 비율:\n", test['label'].value_counts(normalize=True))


from torch.utils.data import Dataset, DataLoader

class DGA_Dataset(Dataset):
    def __init__(self, df):
        self.domains = df['domain'].values
        self.labels = df['label'].values
    
    def __len__(self):
        return len(self.domains)

    def __getitem__(self, idx):
        # Convert: the python list or numpy array to -> a PyTorch tensor
        domain_tensor = torch.LongTensor(self.domains[idx]) # LongTensor는 python list를 포함한 다양한 시퀀스 타입의 데이터를 텐서로 변환할 수 있다.
        label_tensor = torch.LongTensor([self.labels[idx]])
        return domain_tensor, label_tensor
    

# 모델 초기화
from model_mit import MIT
device = torch.device('cuda')
model = MIT().to(device)

# hyperparameters 2, loss function and optimizer from paper 1
batch_size = 100
learning_rate = 1e-3 
epochs = 15 # paper 1에서 12에서 멈췄다.
criterion = nn.BCELoss() # Binary Cross-Entropy Loss 사용
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Adam 옵티마이저 사용

# DataLoader 생성
train_dataset = DGA_Dataset(train)
val_dataset = DGA_Dataset(val)
test_dataset = DGA_Dataset(test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(type(train_loader))

# 학습 및 검증 루프
history = {'val_acc': [], 'train_loss': [], 'val_loss': []}

for epoch in range(epochs):
    # 학습 단계
    model.train() # weight&bias를 update할 수 있는 상태로 변환
    total_loss = 0.0 # loss 초기화
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float()
        
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels.float().view_as(outputs))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)
    history['train_loss'].append(avg_train_loss)

    # 검증 단계
    model.eval()
    correct_predictions = 0
    total_samples = 0
    total_val_loss = 0.0
    
    with torch.no_grad():
        total_val_loss = 0.0 # validation loss 초기화
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()

            outputs = model(inputs)
            predictions = (outputs > 0.5).long() # 학습 시 에폭마다 val acc 출력을 위한 임시 threshold
            
            correct_predictions += (predictions.squeeze() == labels.squeeze()).sum().item()
            total_samples += labels.size(0)

            # val loss도 계산
            val_loss = criterion(outputs, labels.float().view_as(outputs))
            total_val_loss += val_loss.item()
            
        avg_val_loss = total_val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)


    accuracy = (correct_predictions / total_samples) * 100
    history['val_acc'].append(accuracy)


    print(f"Epoch [{epoch+1}/{epochs}] | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Validation Loss: {avg_val_loss:.4f} | "
          f"Validation Accuracy: {accuracy:.2f}%")
    
    # 에포크마다 wandb에 로그 기록
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "val_accuracy": accuracy,
    })

# threshold 정하기
with torch.no_grad():
    model.eval()
    correct_predictions = 0
    total_samples = 0
    
    val_outputs, val_labels = [], []
    for inputs, labels in tqdm.tqdm(val_loader):
        inputs, labels = inputs.to(device), labels.to(device).float()

        outputs = model(inputs)
        false_positive = outputs[labels == 0]

        val_outputs.append(outputs.squeeze().cpu())
        val_labels.append(labels.cpu)

    val_outputs = torch.cat(val_outputs) # append된 텐서들을 하나의 텐서로 합침
    val_labels = torch.cat(val_labels) 

    false_positive = val_outputs[val_labels == 0] # 실제 레이블이 0인 것들 중 모델이 양성(1)으로 예측한 것들 # 이걸 100으로 나누면 FPR이 됨
    threshold = torch.quantile(false_positive, 0.9995).item() # 99.9th percentile

print(f"Chosen threshold (FPR을 0.001로 만드는 임계값: {threshold:.4f}")
wandb.log({"final_threshold": threshold})

with torch.no_grad():
    model.eval()
    correct_predictions = 0
    total_samples = 0

    predictions_table = wandb.Table(columns=["domain_encoded", "label", "prediction", "probability", "is_correct"])

    all_predictions = []
    all_labels = []
    
    for inputs, labels in tqdm.tqdm(test_loader):
        inputs, labels = inputs.to(device), labels.to(device).float()

        outputs = model(inputs)
        predictions = (outputs > threshold).long() # threshold 적용
        
        correct_predictions += (predictions.squeeze() == labels.squeeze()).sum().item()
        total_samples += labels.size(0)

        all_predictions.extend(predictions.squeeze().cpu().tolist())
        all_labels.extend(labels.squeeze().cpu().tolist())

        # 각 배치에 대한 예측 결과와 실제 값을 테이블에 추가
        for i in range(inputs.size(0)):
            domain = inputs[i].cpu().tolist() # 도메인 데이터를 다시 리스트로 변환
            label = labels[i].item()
            prediction = predictions[i].item()
            probability = outputs[i].item()
            is_correct = (prediction == label)
            
            predictions_table.add_data(domain, label, prediction, probability, is_correct)

    test_accuracy = (correct_predictions / total_samples) * 100

    all_predictions = torch.tensor(all_predictions)
    all_labels = torch.tensor(all_labels)
    true_positives = ((all_predictions == 1) & (all_labels == 1)).sum().item()
    false_positives = ((all_predictions == 1) & (all_labels == 0)).sum().item()
    total_positives = (all_labels == 1).sum().item()
    total_negatives = (all_labels == 0).sum().item()
    test_TPR = (true_positives / total_positives) * 100
    test_FPR = (false_positives / total_negatives) * 100

    print(f"Test Accuracy with threshold {threshold:.4f}: {test_accuracy:.4f}%")
    print(f"Test TPR with threshold {threshold:.4f}: {test_TPR:.4f}%")
    print(f"Test FPR with threshold {threshold:.4f}: {test_FPR:.4f}%")

    wandb.log({
        "test_accuracy": test_accuracy,
        "test_TPR": test_TPR,
        "test_FPR": test_FPR,
        "test_predictions": predictions_table # 테스트 예측 결과 테이블 기록
    })

# 모델 저장하기
model_path = f'./models/mit_{timestamp}.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# wandb에 저장된 모델 이름 기록
wandb.log({"saved_model_path": model_path})