import torch
from torch import nn
import math
from torchinfo import summary

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, padding_idx) :
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

        # 초기화
        torch.nn.init.xavier_normal_(self.embedding.weight)

    def forward(self, input) :
        return self.embedding(input)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        P_E = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        _2i = torch.arange(0, d_model, step= 2, dtype=torch.float)
        div_term = torch.exp(_2i * (-math.log(10000.0) / d_model)) 

        P_E[:, 0::2] = torch.sin(pos * div_term)
        P_E[:, 1::2] = torch.cos(pos * div_term)

        self.register_buffer('pe', P_E.unsqueeze(0))
        
    def forward(self, x) :
        seq_len = x.size(1)
        pe_slice = self.pe[:, :seq_len, :]
        return x + pe_slice

class Transformer(nn.Module) :
    def __init__(self, d_model, n_heads, dim_feedforward, num_layers, dropout=0.1) :
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.dropout = nn.Dropout(dropout)   

    def forward(self, x, mask=None) :
        return self.dropout(self.encoder(x, src_key_padding_mask=mask))
    
class MTPHead(nn.Module) :
    def __init__(self, d_model, vocab_size, dropout=0.1) :
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x) :

        x = self.dense(x)
        x = self.gelu(x)
        x = self.dropout(x)
        logtis = self.linear(x)
        return logtis
    
class TPPHead(nn.Module) :
    def __init__(self, d_model, vocab_size, dropout=0.1) :
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x) :

        x = self.dense(x)
        x = self.gelu(x)
        x = self.dropout(x)
        logtis = self.linear(x)
        return logtis
    
class TOVHead(nn.Module):
    def __init__(self, d_model, num_classes=2, dropout=0.1, tov_norm = 'pool'):
        super().__init__()
        self.tov_norm = tov_norm
        if tov_norm == "pool" :
            self.dense = nn.Linear(d_model * 2, d_model * 2)
            self.gelu = nn.GELU()
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(d_model * 2, num_classes)
        else :
            self.dense = nn.Linear(d_model, d_model)
            self.gelu = nn.GELU()
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, sequence_output):

        if self.tov_norm == "pool" :
            max_output = sequence_output.max(dim=1).values
            avg_ouput = sequence_output.mean(dim=1)
            output = torch.cat((max_output, avg_ouput), dim=1)
        else :
            output = sequence_output[:, 0, :]
        
        x = self.dense(output)
        x = self.gelu(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits
    
class PretrainedModel(nn.Module) :
    def __init__(self, vocab_size, d_model, n_heads, dim_feedforward, 
                 num_layers, max_len, dropout=0.1, padding_idx=0, tov_norm='pool') :
        super().__init__()

        self.d_model = d_model
        self.padding_idx = padding_idx
        self.max_len = max_len
        self.tov_norm = tov_norm

        self.embedding = TokenEmbedding(vocab_size, d_model, padding_idx)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.transformer = Transformer(d_model, n_heads, dim_feedforward, num_layers, dropout)

        self.mtp_head = MTPHead(d_model, vocab_size)

        self.tpp_head = TPPHead(d_model, vocab_size)

        self.tov_head = TOVHead(d_model, num_classes=2, dropout=dropout, tov_norm=self.tov_norm)

    def create_padding_mask(self, input_ids):
        return (input_ids == self.padding_idx)
    
    def forward(self, input_ids, task_type='ALL'):
        token_embed = self.embedding(input_ids)
        x = self.positional_encoding(token_embed)
        padding_mask = self.create_padding_mask(input_ids)
        encoder_output = self.transformer(x, mask=padding_mask)

        outputs = {}

        # Task 1: MTP
        if task_type == 'MTP' or task_type == 'ALL':
            outputs['mtp_logits'] = self.mtp_head(encoder_output)
            
        # Task 2: TTP
        if task_type == 'TPP' or task_type == 'ALL':
            outputs['ttp_logits'] = self.tpp_head(encoder_output)
            
        # Task 3: TOV
        if task_type == 'TOV' or task_type == 'ALL':
            outputs['tov_logits'] = self.tov_head(encoder_output)

        # 단일 태스크를 요구했을 경우 dict 대신 로짓 텐서 자체를 반환
        if len(outputs) == 1 and task_type != 'ALL':
            return list(outputs.values())[0]
            
        return outputs


class FinetuningHead(nn.Module) :
    def __init__(self, d_model, dropout, clf_norm = 'pool') :
        super().__init__()

        self.d_model = d_model
        self.clf_norm = clf_norm

        if clf_norm == "pool" :
            self.dense1 = nn.Linear(d_model * 4, d_model * 2)
        else :
            self.dense1 = nn.Linear(d_model * 2, d_model * 2)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model * 2, 2)

    def forward(self, encoder_output) :

        x = self.dense1(encoder_output)
        x = torch.relu(x)
        x = self.dropout(x)

        logits = self.classifier(x)
        return logits
    
class FineTuningModel(nn.Module):
    def __init__(self, pretrain_model_t, pretrain_model_c, dropout=0.1, padding_idx=0, clf_norm = 'pool'):
        super().__init__()

        self.padding_idx = padding_idx
        self.clf_norm = clf_norm

        # --- Token Path Components ---
        # load_pretrain(pretrain_model_t)
        self.transformer_encoder_t = pretrain_model_t.transformer
        self.embedding_t = pretrain_model_t.embedding
        self.positional_encoding_t = pretrain_model_t.positional_encoding

        # --- Character Path Components ---
        # load_pretrain(pretrain_model_c)
        self.transformer_encoder_c = pretrain_model_c.transformer
        self.embedding_c = pretrain_model_c.embedding
        self.positional_encoding_c = pretrain_model_c.positional_encoding
        
        # DGA 분류 헤드 연결
        self.classifier_head = FinetuningHead(
            d_model=pretrain_model_t.d_model,
            dropout=dropout,
            clf_norm=self.clf_norm
        )

    def create_padding_mask(self, input_ids):
        return (input_ids == self.padding_idx)

    def forward(self, input_ids_t, input_ids_c):

        # --- 1. Token Path (X_t) 처리 ---
        token_embed_t = self.embedding_t(input_ids_t)
        x_t = self.positional_encoding_t(token_embed_t)
        padding_mask_t = self.create_padding_mask(input_ids_t)
        
        encoder_output_t = self.transformer_encoder_t(x_t, mask=padding_mask_t)

        if self.clf_norm == 'pool' :
            max_output_t = encoder_output_t.max(dim=1).values
            avg_output_t = encoder_output_t.mean(dim=1)
            cls_output_t = torch.cat((max_output_t, avg_output_t), dim=1)
        else :
            cls_output_t = encoder_output_t[:, 0, :]

        # --- 2. Character Path (X_c) 처리 ---
        token_embed_c = self.embedding_c(input_ids_c)
        x_c = self.positional_encoding_c(token_embed_c)
        padding_mask_c = self.create_padding_mask(input_ids_c)
        
        encoder_output_c = self.transformer_encoder_c(x_c, mask=padding_mask_c)

        if self.clf_norm == 'pool' :
            max_output_c = encoder_output_c.max(dim=1).values
            avg_output_c = encoder_output_c.mean(dim=1)
            cls_output_c = torch.cat((max_output_c, avg_output_c), dim=1)
        else :
            cls_output_c = encoder_output_c[:, 0, :]

        combined_output = torch.cat((cls_output_t, cls_output_c), dim=1)
        
        logits = self.classifier_head(combined_output)
        return logits