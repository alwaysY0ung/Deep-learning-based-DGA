import torch
from torch import nn
import math
from torchinfo import summary

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        torch.nn.init.xavier_normal_(self.embedding.weight)

    def forward(self, input):
        return self.embedding(input)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        P_E = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        _2i = torch.arange(0, d_model, step=2, dtype=torch.float)
        div_term = torch.exp(_2i * (-math.log(10000.0) / d_model)) 

        P_E[:, 0::2] = torch.sin(pos * div_term)
        P_E[:, 1::2] = torch.cos(pos * div_term)

        self.register_buffer('pe', P_E.unsqueeze(0))
        
    def forward(self, x):
        # 입력 길이에 맞춰 slicing
        seq_len = x.size(1)
        pe_slice = self.pe[:, :seq_len, :]
        return x + pe_slice

class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, num_layers, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x, mask=None):
        return self.encoder(x, src_key_padding_mask=mask)
    
class TokenPredictionHead(nn.Module):
    def __init__(self, d_model, vocab_size, dropout=0.1): # 256, 131
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        """
        pretrain 대비 finetuning에서 더 복잡한 문제를 풀 경우 (ex. 사진 분류로 pretrain하다가 finetuning에서 배경에 우주가 추가된 경우)
        finetuning단계에서 loss가 증가하는 경향을 보일 수 있는데
        이때 linear를 1개에서 2개로 증가시키는 것이 loss를 줄이는 방법이 될 수 있다
        자주 도입되는 방법임.
        """
        x = self.linear(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class PositionPredictionHead(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model, max_len) 
        
    def forward(self, x):
        x = self.linear(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class SequenceClassificationHead(nn.Module):
    def __init__(self, d_model, num_classes=2, dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(d_model*2, d_model*2)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model*2, num_classes)

        
    def forward(self, sequence_output): 
        # [CLS] 토큰 벡터 추출 (첫 번째 위치, index 0)
        # print(sequence_output.shape) # 128, 80, 256

        # cls_output = sequence_output[:, -1, :] # 입력된 시퀀스의 가장 첫 번째(인덱스 0) 벡터만 쏙 뽑아서 분류(0인지 1인지)에 사용
        #                                     #BERT류 모델의 약속된 규칙임. 즉, 0번 자리에 [CLS]가 없으면 분류가 제대로 되지 않음.
        
        max = sequence_output.max(dim=1).values
        mean = sequence_output.mean(dim=1)

        cls_output = torch.cat([max,mean], dim=1)

        x = self.dense(cls_output) 
        x = self.gelu(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits
    
class PretrainedModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, dim_feedforward, 
                    num_layers, max_len, dropout=0.1, padding_idx=0):
        super().__init__()

        self.d_model = d_model
        self.padding_idx = padding_idx
        self.max_len = max_len

        self.embedding = TokenEmbedding(vocab_size, d_model, padding_idx)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.transformer = Transformer(d_model, n_heads, dim_feedforward, num_layers, dropout)

        self.mlm_head = TokenPredictionHead(d_model, vocab_size) # 128, 131
        self.permutation_head = PositionPredictionHead(d_model, max_len)
        self.binary_clf_head = SequenceClassificationHead(d_model, num_classes=2, dropout=dropout)

        

    def create_padding_mask(self, input_ids):
        return (input_ids == self.padding_idx)
    
    def forward(self, input_ids, task_type='ALL'):
        token_embed = self.embedding(input_ids)
        x = self.positional_encoding(token_embed)
        padding_mask = self.create_padding_mask(input_ids)
        encoder_output = self.transformer(x, mask=padding_mask)
        # print(encoder_output.shape) # 128, 80, 256

        outputs = {}

        if task_type == 'MLM' or task_type == 'ALL':
            outputs['mlm_logits'] = self.mlm_head(encoder_output)
            
        if task_type == 'PERMUTATION' or task_type == 'ALL':
            outputs['perm_logits'] = self.permutation_head(encoder_output)
            
        if task_type == 'BINARY_CLF' or task_type == 'ALL':
            outputs['binary_logits'] = self.binary_clf_head(encoder_output)

        if len(outputs) == 1 and task_type != 'ALL':
            return list(outputs.values())[0]
            
        return outputs


import torch
from torchinfo import summary

# 1. 모델 인스턴스 생성 (하이퍼파라미터는 상황에 맞게 조정)
vocab_size = 131 # 127 + 4
d_model = 256
n_heads = 8
dim_feedforward = 2048
num_layers = 4
max_len = 80

model = PretrainedModel(vocab_size, d_model, n_heads, dim_feedforward, num_layers, max_len)

# 2. summary 출력
# 배치 사이즈 32, 시퀀스 길이 128인 데이터를 가정
BATCH_SIZE = 32
SEQ_LEN = max_len

# summary(
#     model, 
#     input_size=(BATCH_SIZE, SEQ_LEN), # 입력 데이터의 크기 (Batch, Seq_len)
#     dtypes=[torch.long],              # 임베딩 층 입력을 위해 정수형(Long) 지정 필수
#     col_names=["input_size", "output_size", "num_params"], # 출력할 컬럼 설정
#     depth=3                           # 계층 구조를 얼마나 깊게 보여줄지 설정
# )

summary(model)