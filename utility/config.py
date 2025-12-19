from dataclasses import dataclass

@dataclass
class PretrainConfig:
    # Data
    max_len_char: int = 77
    vocab_size_char: int = 131
    text_col: str = "domain"
    label_col: str = "label"
    
    # Vocabulary (Subword)
    max_len_subword: int = 30
    vocab_size_subword: int = 30522
    min_freq_subword: int = 2
    use_bert_pretokenizer: bool = False
    
    # Model
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 12
    dim_feedforward: int = 768
    dropout: float = 0.1
    
    # Training
    batch_size: int = 128
    num_workers: int = 4
    lr: float = 1e-4
    
    # Pretraining Tasks
    mask_ratio: float = 0.15
    shuffle_prob: float = 0.5
    tov_norm: str = "cls"  # "cls" or "pool"
    ignore_index: int = -100
    
    # Output
    save_path: str = "pretrained.pt"
