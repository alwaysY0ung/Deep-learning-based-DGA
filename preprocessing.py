import torch
from torch.utils.data import Dataset
import numpy as np
import random


class SpecialIDs:
    pad_id: int = 0
    unk_id: int = 1
    cls_id: int = 2
    sep_id: int = 3
    mask_id: int = 4


def mtp_dataset(inputs, special_ids, max_len, mask_ratio=0.15, ignore_idx=-100) :
    labels = np.full(max_len, ignore_idx, dtype=np.int64)
    non_padding_indices = np.where(inputs != special_ids.pad_id)[0]
    non_padding_indices = non_padding_indices[:-1]

    if len(non_padding_indices) <= 1:
        return inputs, labels
    
    # 마스크할 토큰 개수(최소 1개)
    num_mask = max(1, int(len(non_padding_indices) * mask_ratio))

    masked_indices = random.sample(non_padding_indices.tolist(), num_mask)

    masked_inputs = np.copy(inputs)

    for idx in masked_indices :
        labels[idx] = inputs[idx]
        masked_inputs[idx] = special_ids.mask_id

    return masked_inputs, labels


def tpp_dataset(inputs, special_ids, ignore_idx=-100) :
    labels = np.copy(inputs)
    labels[inputs == special_ids.pad_id] = ignore_idx
    non_padding_indices = np.where(inputs != special_ids.pad_id)[0]
    non_padding_indices = non_padding_indices[:-1]

    if len(non_padding_indices) <= 1:
        return inputs, labels
    
    shuffled_inputs = np.copy(inputs)
    shuffled_indices = non_padding_indices.tolist()

    permuted_indices = shuffled_indices.copy()
    random.shuffle(permuted_indices)
    
    for i, original_pos in enumerate(shuffled_indices):
        new_pos = permuted_indices[i]
        shuffled_inputs[new_pos] = inputs[original_pos]

    return shuffled_inputs, labels


def tov_dataset(inputs, special_ids, max_len, shuffle_prob=0.5, tov_norm='cls') :
    non_padding_indices = np.where(inputs != special_ids.pad_id)[0]
    non_padding_indices = non_padding_indices[:-1]
    processed_inputs = np.copy(inputs)

    if len(non_padding_indices) <= 1:
        is_scramble = False
        label = 0
    else:
        is_scramble = random.random() < shuffle_prob
        label = 1 if is_scramble else 0

    if is_scramble :
        shuffled_indices = non_padding_indices.tolist()
        original_values = [inputs[i] for i in shuffled_indices]
        random.shuffle(original_values)

        for i, idx in enumerate(shuffled_indices) :
            processed_inputs[idx] = original_values[i]

    pure_tokens = processed_inputs[non_padding_indices].tolist()

    if len(pure_tokens) > max_len - 2:
        pure_tokens = pure_tokens[:max_len - 2]

    if tov_norm == 'cls' :
        ids = [special_ids.cls_id]
        ids.extend(pure_tokens)
    elif tov_norm == 'pool' :
        ids = pure_tokens

    ids.append(special_ids.sep_id)

    if len(ids) < max_len:
        ids += [special_ids.pad_id] * (max_len - len(ids))

    return ids, label
    

class SubTaskDataset(Dataset) :
    def __init__(self, df, domain_col='domain', label_col='label', max_len=77, mask_ratio=0.15, ignore_idx=-100, shuffle_prob = 0.5,
                tokenizer=None, special_ids=SpecialIDs, type='char', tov_norm='cls'):
        self.df = df
        self.domain_col = domain_col
        self.label_col = label_col
        self.max_len = max_len
        self.mask_ratio = mask_ratio
        self.ignore_idx = ignore_idx
        self.shuffle_prob = shuffle_prob
        self.special_ids = special_ids
        self.pad_idx = special_ids.pad_id
        self.unk_idx = special_ids.unk_id
        self.mask_idx = special_ids.mask_id
        self.cls_idx = special_ids.cls_id
        self.sep_idx = special_ids.sep_id
        self.type = type
        self.tov_norm = tov_norm
        if self.type == 'char' :
            self.char_list = list("abcdefghijklmnopqrstuvwxyz0123456789-.")
            self.special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
            self.all_tokens = self.special_tokens + self.char_list

            self.char2id = {char: idx for idx, char in enumerate(self.all_tokens)}
        self.tokenizer = tokenizer
        if self.type == 'subword' and  self.tokenizer == None :
            raise ValueError("Tokenizer must be required.")
            

    def domain_to_token(self, domain) :
        domain = domain.lower()
        if self.type == 'subword' :
            encoded = self.tokenizer(domain, add_special_tokens=False)
            token_indices = encoded["input_ids"]
        elif self.type == 'char' :
            token_indices = [self.char2id.get(c, self.unk_idx) for c in domain]

        # zero padding
        if len(token_indices) > self.max_len - 1:
            token_indices = token_indices[:self.max_len - 1]
            token_indices.append(self.sep_idx)
        else:
            token_indices.append(self.sep_idx)
            token_indices += [self.pad_idx] * (self.max_len - len(token_indices))
        return np.array(token_indices, dtype=np.int64)

    def mtp(self, inputs) :
        return mtp_dataset(inputs, self.special_ids, self.max_len, self.mask_ratio, self.ignore_idx)

    def tpp(self, inputs) :
        return tpp_dataset(inputs, self.special_ids, self.ignore_idx)

    def tov(self, inputs) :
        return tov_dataset(inputs, self.special_ids, self.max_len, self.shuffle_prob, self.tov_norm)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        domain, _ = self.df.row(idx)
        X_ori = self.domain_to_token(domain)

        # 1. MTP 데이터 생성
        X_mtp, Y_mtp = self.mtp(X_ori)
        
        # 2. TPP 데이터 생성
        X_tpp, Y_tpp = self.tpp(X_ori)
        
        # 3. TOV 데이터 생성
        X_tov, Y_tov = self.tov(X_ori)
        

        # 최종 반환: 6개의 텐서를 튜플로 묶어 반환
        return (torch.tensor(X_mtp, dtype=torch.long), 
                torch.tensor(Y_mtp, dtype=torch.long),
                torch.tensor(X_tpp, dtype=torch.long), 
                torch.tensor(Y_tpp, dtype=torch.long),
                torch.tensor(X_tov, dtype=torch.long), 
                torch.tensor(Y_tov, dtype=torch.long))
  
    
class FineTuningDataset(Dataset) :
    def __init__(self, df, domain_col='domain', label_col='label', special_ids=SpecialIDs, max_len_t=30, max_len_c=77, tokenizer=None, clf_norm='cls'):
        self.df = df
        self.domain_col = domain_col
        self.label_col = label_col
        self.max_len_t = max_len_t
        self.max_len_c = max_len_c
        self.tokenizer = tokenizer
        self.clf_norm = clf_norm
        if tokenizer == None :
            raise ValueError("Tokenizer must be required.")
        self.special_ids = special_ids
        self.pad_idx = special_ids.pad_id
        self.unk_idx = special_ids.unk_id
        self.mask_idx = special_ids.mask_id
        self.cls_idx = special_ids.cls_id
        self.sep_idx = special_ids.sep_id
        self.char_list = list("abcdefghijklmnopqrstuvwxyz0123456789-.")
        self.special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
        self.all_tokens = self.special_tokens + self.char_list

        self.char2id = {char: idx for idx, char in enumerate(self.all_tokens)}
        self.id2char = {idx: char for idx, char in enumerate(self.all_tokens)}

    def domain_to_ids(self, domain):
        domain = domain.lower()
        
        token_indices = [self.char2id.get(c, self.unk_idx) for c in domain]

        if len(token_indices) > self.max_len_c - 1:
            token_indices = token_indices[:self.max_len_c - 1]

        if self.clf_norm == 'cls' :
            ids = [self.cls_idx]
            ids.extend(token_indices)
        elif self.clf_norm == 'pool' :
            ids = token_indices

        ids.append(self.sep_idx)

        if len(ids) < self.max_len_c:
            ids += [self.pad_idx] * (self.max_len_c - len(ids))
            
        return np.array(ids, dtype=np.int64)
    
    def domain_to_token(self, domain) :
        domain = domain.lower()
        encoded = self.tokenizer(domain, add_special_tokens=False)
        token_indices = encoded["input_ids"]

        if len(token_indices) > self.max_len_t - 1:
            token_indices = token_indices[:self.max_len_t - 1]

        if self.clf_norm == 'cls' :
            ids = [self.cls_idx]
            ids.extend(token_indices)
        elif self.clf_norm == 'pool' :
            ids = token_indices

        ids.append(self.sep_idx)

        if len(ids) < self.max_len_t:
            ids += [self.pad_idx] * (self.max_len_t - len(ids))
        
        return np.array(ids, dtype=np.int64)
    
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        domain, label = self.df.row(idx)
        X_token = self.domain_to_token(domain)
        X_char = self.domain_to_ids(domain)
        y = np.int64(label)
        return torch.tensor(X_token, dtype=torch.long), torch.tensor(X_char, dtype=torch.long), torch.tensor(y, dtype=torch.long)

if __name__ == '__main__':

    from transformers import AutoTokenizer
    import polars as pl

    df = pl.DataFrame({'domain': ['a'], 'label': [1]})

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    dataset = SubTaskDataset(df, tokenizer=tokenizer, max_len=30, type='subword', tov_norm='cls')

    X_mtp, Y_mtp, X_tpp, Y_tpp, X_tov, Y_tov = dataset[0]

    print(f"\n도메인 원본: {df.row(0)[0]}")
    print(f"전처리된 X: {X_mtp.shape}")
    print(f"X MTP(앞 20개): {X_mtp[:20].tolist()}")
    print(f"X TPP(앞 20개): {X_tpp[:20].tolist()}")
    print(f'X TOV(앞 20개): {X_tov[:20].tolist()}')
    print(f'Y MTP(앞 20개): {Y_mtp[:20].tolist()}')

    dataset = FineTuningDataset(df, tokenizer=tokenizer, clf_norm='cls')

    X_token, X_char, y = dataset[0]

    print("도메인 원본:", df.row(0)[0])
    print("전처리된 X (길이, 토큰):", X_token.shape) 
    print("전처리된 X (앞 20개, 토큰):", X_token[:20].tolist())
    print("전처리된 X (길이, 문자):", X_char.shape) 
    print("전처리된 X (앞 20개, 문자):", X_char[:20].tolist())
    print("라벨 y:", y.item())