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
from utility.dataset import get_train_set_tld

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

def train(file_paths,
        text_col: str,
        vocab_size: int,
        min_freq: int,
        use_bert_pretokenizer: bool = True,
        save_path : str = "tokenizer-{min_freq}-{vocab_size}-tld.json"
    ) -> Tokenizer:

    print("[.tld] 추출 중...")
    tld_df = (
        pl.scan_parquet(file_paths)  # 리스트를 넣으면 Polars가 모든 파일을 스캔
        .select(pl.col(text_col).str.extract_all(r"\[\.[a-zA-Z0-9-]+\]"))
        .explode(text_col)
        .unique()
        .drop_nulls()
        .collect() # 최종 결과를 DataFrame으로 가져옴
    )
    tld_special_tokens = tld_df[text_col].to_list()
    print(f"Unique TLD tokens 발견: {len(tld_special_tokens)}")

    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.Lowercase(),
        normalizers.StripAccents()])

    if use_bert_pretokenizer:
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

    # Token IDs: [PAD]:0, [UNK]:1, [CLS]:2, [SEP]:3, [MASK]:4, [TRUNC]:5, TLDs:6~ (순서대로 할당)
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[TRUNC]"] + tld_special_tokens
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size, # = 30522(bert) + 1 + len(tld_special_tokens)
        min_frequency=min_freq,
        special_tokens=special_tokens)

    # training
    corpus_iter = get_corpus_batches(file_paths=file_paths, column=text_col)
    tokenizer.train_from_iterator(corpus_iter, trainer=trainer)

    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS]:0 $A:0 [SEP]:0",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
            ("[PAD]", tokenizer.token_to_id("[PAD]")),
        ],
    )

    tokenizer.save(str(save_path))
    return tokenizer

if __name__ == "__main__":
    cfg = PretrainConfig
    vocab = cfg.vocab_size_subword
    minfreq = cfg.min_freq_subword
    _, paths = get_train_set_tld()

    trained_tokenizer = train(
        file_paths=paths,
        text_col="domain",
        vocab_size=vocab,
        min_freq=minfreq,
        use_bert_pretokenizer=True
    )

    # 검증: [.com]이 하나의 토큰으로 나오는지 확인
    test_domain = "google[.com]"
    encoded = trained_tokenizer.encode(test_domain)
    print(f"Test Sentence: {test_domain}")
    print(f"Tokens: {encoded.tokens}")
    print(f"Token IDs: {encoded.ids}")
    # 예상 결과: ['[CLS]', 'google', '[.com]', '[SEP]']