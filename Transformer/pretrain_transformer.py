import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_processor_char import SubTaskDataset, PAD_IDX, MASK_IDX, CLS_IDX, SEP_IDX, CHAR_OFFSET
from model_transformer import PretrainedModel
from tqdm import tqdm
import datetime
import wandb

def train_multitask(model, dataset, val_dataset, optimizer, device, num_epochs, batch_size, ignore_idx):
    best_val_loss = float('inf')
    best_epoch = 0

    criterion_token = nn.CrossEntropyLoss(ignore_index=ignore_idx)
    criterion_binary = nn.CrossEntropyLoss()

    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_mlm_loss = 0
        total_perm_loss = 0
        total_bin_loss = 0

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        train_loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)

        for X_mlm, Y_mlm, X_perm, Y_perm, X_bin, Y_bin in train_loop:
            optimizer.zero_grad()

            X_mlm, Y_mlm = X_mlm.to(device), Y_mlm.to(device)
            X_perm, Y_perm = X_perm.to(device), Y_perm.to(device)
            X_bin, Y_bin = X_bin.to(device), Y_bin.to(device)

            # --- T1: MLM Loss ---
            logits_mlm = model(X_mlm, task_type='MLM')
            loss_mlm = criterion_token(logits_mlm.view(-1, logits_mlm.size(-1)), Y_mlm.view(-1))

            # --- T2: PERMUTATION Loss ---
            logits_perm = model(X_perm, task_type='PERMUTATION')
            loss_perm = criterion_token(logits_perm.view(-1, logits_perm.size(-1)), Y_perm.view(-1))

            # --- T3: BINARY_CLF Loss ---
            logits_bin = model(X_bin, task_type='BINARY_CLF')
            loss_bin = criterion_binary(logits_bin, Y_bin)

            L_total = loss_mlm + loss_perm + loss_bin
            L_total.backward()
            optimizer.step()

            total_mlm_loss += loss_mlm.item()
            total_perm_loss += loss_perm.item()
            total_bin_loss += loss_bin.item()

            current_step = train_loop.n + 1 # 현재 완료된 배치의 수를 반환하므로, +1을 하면 현재의 배치번째가 나옴 (1~N번째)
            avg_total = (total_mlm_loss + total_perm_loss + total_bin_loss) / current_step
            
            train_loop.set_postfix(total=f'{avg_total:.4f}', mlm=f'{total_mlm_loss/current_step:.4f}', 
                                    perm=f'{total_perm_loss/current_step:.4f}', bin=f'{total_bin_loss/current_step:.4f}')

        # --- Train Epoch 평균 계산 (WandB용) ---
        avg_train_mlm = total_mlm_loss / len(dataloader) # len(dataloader) 연산: 전체데이터개수 / 배치크기 (이때 기본적으로 소수점은 올림 처리하여 마지막 자투리 데이터까지 포함.)
        avg_train_perm = total_perm_loss / len(dataloader) # 즉 전체 step수로 나눠주면 되는 것
        avg_train_bin = total_bin_loss / len(dataloader) # 위에 for문에서 step 루프를 돌지만 for문 밖에서 train_loop.n을 할 수 없으니까 len(dataloader)로 구하는 것
        avg_train_total = avg_train_mlm + avg_train_perm + avg_train_bin
        
        # Validation Logic
        val_total, val_mlm, val_perm, val_bin = evaluate_multitask(model, val_dataset, device, batch_size, ignore_idx)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_total:.4f}, Val Loss: {val_total:.4f}")

        # --- [수정 1] WandB Logging (8개 지표) ---
        wandb.log({
            "train/total_loss": avg_train_total,
            "train/mlm_loss": avg_train_mlm,
            "train/perm_loss": avg_train_perm,
            "train/bin_loss": avg_train_bin,
            "val/total_loss": val_total,
            "val/mlm_loss": val_mlm,
            "val/perm_loss": val_perm,
            "val/bin_loss": val_bin
        })

        # --- [수정 2] 모델 저장 로직 ---
        
        # 1. Best Model 저장
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f"./Transformer/model/{best_filename}_best.pt")
            print(f"  [*] Best model saved (Epoch {best_epoch})")

        # 2. Every Epoch 저장
        torch.save(model.state_dict(), f"./Transformer/model/{best_filename}_epoch_{epoch+1}.pt")

    return best_epoch

def evaluate_multitask(model, dataset, device, batch_size, ignore_idx):
    criterion_token = nn.CrossEntropyLoss(ignore_index=ignore_idx)
    criterion_binary = nn.CrossEntropyLoss()
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    sum_mlm_loss = 0
    sum_perm_loss = 0
    sum_bin_loss = 0
    
    with torch.no_grad():
        for X_mlm, Y_mlm, X_perm, Y_perm, X_bin, Y_bin in dataloader:
            X_mlm, Y_mlm = X_mlm.to(device), Y_mlm.to(device)
            X_perm, Y_perm = X_perm.to(device), Y_perm.to(device)
            X_bin, Y_bin = X_bin.to(device), Y_bin.to(device)

            logits_mlm = model(X_mlm, 'MLM')
            l1 = criterion_token(logits_mlm.view(-1, logits_mlm.size(-1)), Y_mlm.view(-1))
            sum_mlm_loss += l1.item() # 누적

            logits_perm = model(X_perm, 'PERMUTATION')
            l2 = criterion_token(logits_perm.view(-1, logits_perm.size(-1)), Y_perm.view(-1))
            sum_perm_loss += l2.item() # 누적

            logits_bin = model(X_bin, 'BINARY_CLF')
            l3 = criterion_binary(logits_bin, Y_bin)
            sum_bin_loss += l3.item() # 누적
            
    total_steps  = len(dataloader)
    avg_mlm = sum_mlm_loss / total_steps # total_steps = num_batches != batch_size
    avg_perm = sum_perm_loss / total_steps 
    avg_bin = sum_bin_loss / total_steps
    avg_total = avg_mlm + avg_perm + avg_bin
            
    return avg_total, avg_mlm, avg_perm, avg_bin

if __name__ == '__main__':
    # --- 설정 ---
    D_MODEL = 256
    N_HEADS = 8
    MAX_LEN = 80
    DIM_FEEDFORWARD = 2048
    NUM_LAYERS = 4
    
    VOCAB_SIZE_FINAL = 131 # 127 + 4

    NUM_EPOCHS = 100
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    MLM_PROB = 0.15
    IGNORE_IDX = -100
    SAMPLING_RATE_train = False # 0.1 # 사용 안 할 시 False로 바꿔줘야
    SAMPLING_RATE_val = False # 0.5
    SAMPLING = bool(SAMPLING_RATE_train and SAMPLING_RATE_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_filename = f"char_model_{datetime.datetime.now().strftime('%m%d_%H%M')}"
    
    wandb.init(project='proposal', name=best_filename, config={
        "type": "Character-Level Transformer",
        "d_model": D_MODEL,
        "vocab_size": VOCAB_SIZE_FINAL,
        "SAMPLING" : SAMPLING, # True of False가 들어감
    })
    
    # 데이터 로드 (경로는 환경에 맞게 수정)
    train_df1 = pd.read_parquet('./cache/period_data/T17_benign_train.parquet')
    train_df2 = pd.read_parquet('./cache/period_data/T18_benign_train.parquet')
    val_df1 = pd.read_parquet('./cache/period_data/T17_benign_val.parquet')
    val_df2 = pd.read_parquet('./cache/period_data/T18_benign_val.parquet')

    train_df = pd.concat([train_df1, train_df2], ignore_index=True)
    val_df = pd.concat([val_df1, val_df2], ignore_index=True)
    
    # # 돌아가는 거 테스트용 더미 데이터
    # train_df = pd.DataFrame({'domain': ['google.com', 'naver.com']*100, 'label': [0,0]*100})
    # val_df = pd.DataFrame({'domain': ['test.com', 'example.com']*10, 'label': [0,0]*10})

    print(f"Original Train Size: {len(train_df)}")

    if SAMPLING:
        print(f"train set sampling rate: {SAMPLING_RATE_train}")
        print(f"val set sampling rate: {SAMPLING_RATE_val}")
        # ==========================================
        # 랜덤 샘플링 추가
        # frac=0.1은 10%를 의미 / random_state를 고정해야 매번 같은 데이터로 실험 가능
        train_df = train_df.sample(frac=SAMPLING_RATE_train, random_state=42).reset_index(drop=True) # 15만
        val_df = val_df.sample(frac=SAMPLING_RATE_val, random_state=42).reset_index(drop=True) # 7.5만
        # ==========================================

    
    train_dataset = SubTaskDataset(train_df, max_len=MAX_LEN, prob=MLM_PROB, ignore_idx=IGNORE_IDX)
    val_dataset = SubTaskDataset(val_df, max_len=MAX_LEN, prob=MLM_PROB, ignore_idx=IGNORE_IDX)

    print(f"final Train Size: {len(train_df)}")
    print(f"final Val Size: {len(val_df)}")

    model = PretrainedModel(
        vocab_size=VOCAB_SIZE_FINAL,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        dim_feedforward=DIM_FEEDFORWARD,
        num_layers=NUM_LAYERS,
        max_len=MAX_LEN,
        padding_idx=PAD_IDX # 0
    )

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_multitask(model, train_dataset, val_dataset, optimizer, device, NUM_EPOCHS, BATCH_SIZE, IGNORE_IDX)
    wandb.finish()