import polars as pl
import argparse
import glob
from pathlib import Path
from utility.config import PretrainConfig
from tokenizers import (
    Tokenizer,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers
)

def get_corpus_batches(file_paths, column="domain", batch_size=10000):
    """
    Parquet 파일들을 Lazy하게 스캔하여 중복을 제거한 후,
    Tokenizer 학습을 위해 배치 단위로 yield 합니다."""

    q = pl.scan_parquet(file_paths)
    q = q.select(column).unique().drop_nulls()
    df_unique = q.collect()
    
    total_rows = df_unique.height

    for i in range(0, total_rows, batch_size):
        yield df_unique[column].slice(i, batch_size).to_list()

def train(df: pl.DataFrame,
        text_col: str,
        vocab_size: int,
        min_freq: int,
        max_len: int,
        tokenizer_path: str | Path = "artifacts/tokenizer/tokenizer-{min_freq}-{vocab_size}.json",
        use_bert_pretokenizer: bool = False,
    ) -> Tokenizer:

    corpus_iter = get_corpus_batches(files)
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.Lowercase(),
        normalizers.StripAccents()])

    if use_bert_pretokenizer:
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=special_tokens)

    tokenizer.train_from_iterator(corpus_iter, trainer=trainer)

    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS]:0 $A:0 [SEP]:0",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
            ("[PAD]", tokenizer.token_to_id("[PAD]")),
        ],
    )

    save_path = f"tokenizer-{min_freq}-{vocab_size}.json"
    tokenizer.save(save_path)

if __name__ == "__main__":
    files = ["./../cache/period_data_no_underscore/T17_benign.parquet",
                "./../cache/period_data_no_underscore/T18_benign.parquet",
                "./../cache/period_data_no_underscore/T19_benign.parquet",
                "./../cache/period_data_no_underscore/T17_dga.parquet",
                "./../cache/period_data_no_underscore/T18_dga.parquet",
                "./../cache/period_data_no_underscore/T19_dga.parquet"
                ]

    cfg = PretrainConfig
    vocab = cfg.vocab_size_subword
    minfreq = cfg.min_freq_subword

    train(vocab, minfreq)