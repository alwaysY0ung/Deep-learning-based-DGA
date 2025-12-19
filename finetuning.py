import pandas as pd
import polars as pl
pl.Config.set_engine_affinity(engine="streaming")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocessing import FineTuningDataset
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast
from model import PretrainedModel, FineTuningModel
from tqdm import tqdm
import time
import datetime
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score
from utility import dataset
from utility.path import path_tokenizer, path_model
from utility.config import FinetuningConfig

def load_pretrain_weights(pt_model, weigths_path, device) :
    state_dict = torch.load(weigths_path, map_location=device)

    weights_to_load = {}

    for name, param in state_dict.items() :
        if name.startswith('transformer.') or name.startswith('embedding.') or name.startswith('positional_encoding.') :
            weights_to_load[name] = param

    pt_model.load_state_dict(weights_to_load, strict=False)
    return pt_model

def fine_tune_dga_classifier(pt_model_t, pt_model_c,
                             train_dataloader, val_dataloader, weights_path_t, weights_path_c,
                             device, num_epochs, log_interval_steps, save_path,
                             use_token=True, use_char=True, freeze_backbone=True,
                             unfreeze_at_epoch=0.5,
                             learning_rate=1e-4, backbone_lr=1e-6,):

    pt_t = load_pretrain_weights(pt_model_t, weights_path_t, device) if use_token else None
    pt_c = load_pretrain_weights(pt_model_c, weights_path_c, device) if use_char else None
    
    ft_model = FineTuningModel(
        pretrain_model_t=pt_t, 
        pretrain_model_c=pt_c, 
        freeze_backbone=freeze_backbone,
        clf_norm='pool' # or 'cls' method
    ).to(device)

    trainable_params = [p for p in ft_model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    unfreeze_step = int(len(train_dataloader) * unfreeze_at_epoch) if unfreeze_at_epoch is not None else float('inf')
    backbone_unfrozen = False
    best_val_loss = float('inf')
    global_step = 0 # 전체 스텝 카운터
    
    interval_loss_sum_total = 0 # 로깅 인터벌 동안의 손실 누적합
    interval_batch_counter = 0 # 로깅 인터벌 동안의 배치 카운터

    for epoch in range(num_epochs) :
        ft_model.train()
        total_loss = 0
        train_loop = tqdm(train_dataloader, desc=f'FineTune Epoch {epoch+1}', 
                        bar_format="{l_bar}{n_fmt}/{total_fmt} | [{elapsed}<{remaining} {postfix}]",
                        leave=False)

        for X_token, X_char, y_train in train_loop :
            global_step += 1

            if unfreeze_at_epoch is not None and not backbone_unfrozen and global_step >= unfreeze_step:
                for param in ft_model.parameters(): # 모든 파라미터 해제
                    param.requires_grad = True
                
                param_groups = []
                if ft_model.use_token:
                    param_groups.extend([
                        {'params': ft_model.transformer_encoder_t.parameters(), 'lr': backbone_lr},
                        {'params': ft_model.embedding_t.parameters(), 'lr': backbone_lr},
                        {'params': ft_model.positional_encoding_t.parameters(), 'lr': backbone_lr}
                    ])
                if ft_model.use_char:
                    param_groups.extend([
                        {'params': ft_model.transformer_encoder_c.parameters(), 'lr': backbone_lr},
                        {'params': ft_model.embedding_c.parameters(), 'lr': backbone_lr},
                        {'params': ft_model.positional_encoding_c.parameters(), 'lr': backbone_lr}
                    ])
                param_groups.append({'params': ft_model.classifier_head.parameters(), 'lr': learning_rate})
                
                optimizer = optim.Adam(param_groups)
                backbone_unfrozen = True

            X_token, X_char, y_train = X_token.to(device), X_char.to(device), y_train.to(device)
            optimizer.zero_grad()

            logits = ft_model(
                X_token if use_token else None, 
                X_char if use_char else None
            )
            
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

            current_lr = optimizer.param_groups[0]['lr']

            total_loss += loss.item()

            interval_loss_sum_total += loss.item()
            interval_batch_counter += 1

            if interval_batch_counter == log_interval_steps :

                avg_total_interval_loss = interval_loss_sum_total / interval_batch_counter


                if global_step % 10000 == 0 :

                    avg_val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_finetuning(ft_model, val_dataloader, device)

                    wandb.log({
                        'step': global_step,
                        'interval_train_loss' : avg_total_interval_loss,
                        'val_loss': avg_val_loss,
                        'val_accuracy' : val_acc,
                        'val_precision': val_precision,
                        'val_recall' : val_recall,
                        'val_f1': val_f1,
                        'learning_rate': current_lr
                    })

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        torch.save(ft_model.state_dict(), save_path)

                    train_loop.write(f"[Step {global_step} Interval Log]: Train Loss: {avg_total_interval_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f},"
                        f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
                else :
                    wandb.log({
                        'step': global_step,
                        'interval_train_loss' : avg_total_interval_loss,
                        'learning_rate': current_lr
                    })

                interval_loss_sum_total = 0 
                interval_batch_counter = 0

            current_step = train_loop.n + 1
            avg_total = total_loss / current_step

            train_loop.set_postfix(avg_loss=f'{avg_total:.4f}', refresh=False)

            if global_step % 20000 == 0:
                break

def evaluate_finetuning(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0

    val_preds = []
    val_labels = []

    val_loop = tqdm(dataloader, desc="Validation", disable=True,
                    bar_format="{l_bar}{n_fmt}/{total_fmt} | [{elapsed}<{remaining} {postfix}]",
                    leave=False)

    with torch.no_grad():
        for X_token, X_char, y_val in val_loop:
            X_token, X_char, y_val = X_token.to(device), X_char.to(device), y_val.to(device)
            
            logits = model(
                X_token if model.use_token else None, 
                X_char if model.use_char else None
            )
            
            loss = criterion(logits, y_val)
            total_loss += loss.item()

            predicted_labels = torch.argmax(logits, dim=1)

            val_preds.append(predicted_labels.cpu())
            val_labels.append(y_val.cpu())

            current_step = val_loop.n + 1
            avg_total = total_loss / current_step

            val_loop.set_postfix(avg_loss=f'{avg_total:.4f}', refresh=False)

    avg_loss = total_loss / len(dataloader)

    val_preds = torch.cat(val_preds)
    val_labels = torch.cat(val_labels)

    accuracy = (val_preds == val_labels).sum().item() / len(val_labels)
    precision = precision_score(val_labels, val_preds, zero_division=0)
    recall = recall_score(val_labels, val_preds, zero_division=0)
    f1 = f1_score(val_labels, val_preds, zero_division=0)

    model.train()
    return avg_loss, accuracy, precision, recall, f1

def main():
    cfg = FinetuningConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer_m = PreTrainedTokenizerFast(tokenizer_file=str(path_tokenizer.joinpath(cfg.tokenizer_path)))
    cfg.vocab_size_token = tokenizer_m.vocab_size

    save_dir = path_model.joinpath(cfg.timestamp)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir.joinpath(f"{cfg.best_filename}.pt")

    wandb.init(project=cfg.project_name, name=cfg.best_filename, 
               config=cfg.__dict__, mode=cfg.wandb_mode, tags=['valid'])

    train_df = dataset.get_train_set()
    val_df = dataset.get_val_set()

    train_dataset = FineTuningDataset(train_df, tokenizer=tokenizer_m, 
                                      max_len_t=cfg.max_len_token, max_len_c=cfg.max_len_char)
    val_dataset = FineTuningDataset(val_df, tokenizer=tokenizer_m, 
                                    max_len_t=cfg.max_len_token, max_len_c=cfg.max_len_char)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, 
                                  shuffle=True, num_workers=cfg.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, 
                                shuffle=False, num_workers=cfg.num_workers)

    pt_model_t = PretrainedModel(vocab_size=cfg.vocab_size_token, d_model=cfg.d_model, 
                                 n_heads=cfg.nhead, dim_feedforward=cfg.dim_feedforward, 
                                 num_layers=cfg.num_layers, max_len=cfg.max_len_token)
    pt_model_c = PretrainedModel(vocab_size=cfg.vocab_size_char, d_model=cfg.d_model, 
                                 n_heads=cfg.nhead, dim_feedforward=cfg.dim_feedforward, 
                                 num_layers=cfg.num_layers, max_len=cfg.max_len_char)
    
    save_dir = path_model.joinpath(cfg.timestamp)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir.joinpath(f"{cfg.best_filename}.pt")

    wandb.init(project=cfg.project_name, name=cfg.best_filename, 
               config=cfg.__dict__, mode='online', tags=['valid'])
    
    fine_tune_dga_classifier(
        pt_model_t,
        pt_model_c,
        train_dataloader,
        val_dataloader,
        weights_path_t=path_model.joinpath(cfg.token_weights_path),
        weights_path_c=path_model.joinpath(cfg.char_weights_path),
        device=device,
        num_epochs=cfg.num_epochs,
        log_interval_steps=cfg.log_interval_steps,
        save_path=best_model_path,
        use_token=cfg.use_token,
        use_char=cfg.use_char,
        freeze_backbone=cfg.freeze_backbone,
        learning_rate=cfg.learning_rate,
        backbone_lr=cfg.backbone_lr
    )
    
    if best_model_path.exists():
        artifact = wandb.Artifact(name=cfg.best_filename, type="model")
        artifact.add_file(str(best_model_path))
        wandb.log_artifact(artifact)

    wandb.finish()

if __name__ == '__main__':
    main()