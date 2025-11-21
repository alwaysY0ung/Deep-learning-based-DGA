import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random

# Character Level Constants
PAD_IDX = 0
MASK_IDX = 1
CLS_IDX = 2
SEP_IDX = 3
# 일반 문자는 4번부터 시작 (ASCII + 4)
CHAR_OFFSET = 4

class SubTaskDataset(Dataset):
    def __init__(self, df, domain_col='domain', label_col='label', max_len=75, prob=0.15, ignore_idx=-100):
        self.df = df.reset_index(drop=True)
        self.domain_col = domain_col
        self.label_col = label_col
        self.max_len = 80  # 77(char) + 1(CLS) + 여유분 = 80
        self.prob = prob
        self.ignore_idx = ignore_idx
        
        # Special Token Indices
        self.pad_token_idx = PAD_IDX
        self.mask_token_idx = MASK_IDX
        self.cls_token_idx = CLS_IDX
        self.sep_token_idx = SEP_IDX

    def domain_to_ids(self, domain):
        """
        문자열 -> Character ID 리스트 변환
        Logic: .str[-max_len:] (오른쪽 끝 기준 자르기) -> Left Padding (zfill 효과)
        """
        domain = str(domain).lower()
        
        # 1. Character -> Integer (Offset 적용)
        # ASCII 범위 0~127 가정 (필요시 확장 가능)
        token_indices = [ord(c) + CHAR_OFFSET for c in domain]

        # 2. Truncate
        if len(token_indices) > self.max_len:
            token_indices = token_indices[:self.max_len]

        # 3. Left Padding (zfill과 동일한 효과, 하지만 0 문자 대신 PAD_IDX 사용)
        if len(token_indices) < self.max_len:
            pad_len = self.max_len - len(token_indices)
            token_indices = [self.pad_token_idx] * pad_len + token_indices
            
        return np.array(token_indices, dtype=np.int64)

    def mask_tokens(self, inputs):
        labels = np.full(self.max_len, self.ignore_idx, dtype=np.int64)
        # PAD가 아닌 인덱스 찾기
        non_padding_indices = np.where(inputs != self.pad_token_idx)[0]

        if len(non_padding_indices) <= 1:
            return inputs, labels
        
        num_mask = max(1, int(len(non_padding_indices) * self.prob))
        masked_indices = random.sample(non_padding_indices.tolist(), num_mask)

        masked_inputs = np.copy(inputs)

        for idx in masked_indices:
            labels[idx] = inputs[idx]
            masked_inputs[idx] = self.mask_token_idx

        return masked_inputs, labels
    
    def scramble_tokens(self, inputs):
        labels = np.full(self.max_len, self.ignore_idx, dtype=np.int64)
        non_padding_indices = np.where(inputs != self.pad_token_idx)[0]

        if len(non_padding_indices) <= 1:
            return inputs, labels
        
        scrambled_inputs = np.copy(inputs)
        scramble_indices = non_padding_indices.tolist()

        # 현재 값 -> 위치 매핑 (Permutation 로직 유지)
        # 주의: 같은 문자가 여러 개일 경우 로직이 복잡해질 수 있으나, 기존 로직을 유지.
        # Character 단위에서는 중복 문자(e.g., 'google')가 많으므로
        # 값 기준 매핑보다는 인덱스 자체를 섞는 것이 안전.
        
        # 인덱스 셔플
        permuted_indices = scramble_indices.copy()
        random.shuffle(permuted_indices)
        
        for i, original_pos in enumerate(scramble_indices):
            new_pos = permuted_indices[i]
            
            # 원본 위치의 값을 새 위치로 이동
            scrambled_inputs[new_pos] = inputs[original_pos]
            
            # 레이블: 이 위치(new_pos)에 온 문자가 원래 어디(original_pos) 있었는지
            labels[new_pos] = original_pos 

        return scrambled_inputs, labels
    
    def scramble_or_not(self, inputs):
        """
        Binary Classification용 데이터 생성
        """
        processed_inputs = np.copy(inputs)
        non_padding_indices = np.where(inputs != self.pad_token_idx)[0]

        if len(non_padding_indices) <= 1:
            is_scramble = False
            label = 0
        else:
            is_scramble = random.random() < 0.5
            label = 1 if is_scramble else 0

        if is_scramble:
            scramble_indices = non_padding_indices.tolist()
            original_values = [inputs[i] for i in scramble_indices]
            random.shuffle(original_values)
            for i, idx in enumerate(scramble_indices):
                processed_inputs[idx] = original_values[i]

        # 순수 문자열 추출 (기존 패딩 제외)
        pure_tokens = processed_inputs[non_padding_indices].tolist()
        
        # [CLS] + 문자열
        # 최대 길이(77) + [CLS](1) < max_len(80) 이므로 무조건 패딩만 발생
        combined_tokens = [self.cls_token_idx] + pure_tokens # 길이 상관 X
        
        # 남은 공간 계산 (max_len - 현재길이)
        pad_len = self.max_len - len(combined_tokens)
        
        if pad_len < 0:
            # 만약 80을 넘는 이상치 데이터가 있다면 (없는 거 확인했으나 새로운 test에 있을 수도 있으니까) CLS 보호를 위해 자름
            final_input_ids = combined_tokens[:self.max_len]
        else:
            final_input_ids = ([self.pad_token_idx] * pad_len) + pure_tokens + [self.cls_token_idx] 

        return np.array(final_input_ids, dtype=np.int64), label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        domain = self.df.loc[idx, self.domain_col]
        
        # Base Preprocessing (Left Padded Character IDs)
        X_ori = self.domain_to_ids(domain)

        # 1. MLM
        X_mlm, Y_mlm = self.mask_tokens(X_ori)
        
        # 2. PERMUTATION
        X_perm, Y_perm = self.scramble_tokens(X_ori)
        
        # 3. BINARY_CLF
        X_bin, Y_bin = self.scramble_or_not(X_ori)

        return (torch.tensor(X_mlm, dtype=torch.long), 
                torch.tensor(Y_mlm, dtype=torch.long),
                torch.tensor(X_perm, dtype=torch.long), 
                torch.tensor(Y_perm, dtype=torch.long),
                torch.tensor(X_bin, dtype=torch.long), 
                torch.tensor(Y_bin, dtype=torch.long))

if __name__ == '__main__':
    df = pd.DataFrame({'domain': ['google.com', 'abc'], 'label': [1, 0]})
    dataset = SubTaskDataset(df, max_len=10, prob=0.15) # 짧은 길이로 테스트

    print(f"Max Len: 10, Padding: Left (zfill style)")
    for i in range(2):
        X_mlm, Y_mlm, X_perm, Y_perm, X_bin, Y_bin = dataset[i]
        print(f"\n--- Sample {i} ({df.loc[i, 'domain']}) ---")
        print(f"X_mlm (Left Pad): {X_mlm.tolist()}")
        print(f"X_bin ([CLS] at 0): {X_bin.tolist()}")