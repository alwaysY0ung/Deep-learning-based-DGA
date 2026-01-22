import argparse
import polars as pl
pl.Config.set_engine_affinity(engine="streaming")
import numpy as np
import torch
import time
from preprocessing import FineTuningDataset
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast
from torch.utils.data import DataLoader
from model import PretrainedModel, FineTuningModel
from tqdm import tqdm
import pathlib
import glob
import wandb
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from utility.dataset import get_test_set_20_tld, get_test_set_21_tld, get_test_set_22_tld, get_test_set_23_tld, get_test_set_24_tld, get_test_set_25_tld
from utility.config import PretrainConfig
from utility.path import path_dga_scheme, path_model,path_tokenizer, path_figure 


def compute_metrics(y_true, y_pred):
    metrics = {}

    accuracy = (y_true == y_pred).float().mean().item()
    metrics["accuracy"] = accuracy

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    metrics["tn"] = tn
    metrics["fp"] = fp
    metrics["fn"] = fn
    metrics["tp"] = tp

    # FPR / FNR
    metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    metrics["fnr"] = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    # Precision / Recall / F1
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    return metrics


def test_finetuning(model, device, test_dataloader, use_bert=False):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            batch = [b.to(device) for b in batch]
            y = batch[-1]
            
            if use_bert:
                X_token, X_char, X_bert, X_bert_mask = batch[:-1]
                logits = model(X_token, X_char, X_bert, X_bert_mask)
            else:
                X_token, X_char = batch[:-1]
                logits = model(X_token, X_char)
            
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    metrics = compute_metrics(all_labels, all_preds)

    return metrics, all_preds, all_labels


def test_by_year(cfg, args, model, tokenizer, device, use_bert=False):
    years = [20, 21, 22, 23, 24, 25]

    acc_all, pre_all, rec_all, f1_all, fpr_all, fnr_all = [], [], [], [], [], []
    year_str = []

    global_preds = []
    global_labels = []


    for year in years:
        if year in [24, 25]:
            test_df = get_test_set_24_tld() if year == 24 else get_test_set_25_tld()
        else:
            test_df = (
                get_test_set_20_tld() if year == 20 else
                get_test_set_21_tld() if year == 21 else
                get_test_set_22_tld() if year == 22 else
                get_test_set_23_tld()
            )

        dataset = FineTuningDataset(
            test_df,
            tokenizer=tokenizer,
            max_len_t=cfg.max_len_subword,
            max_len_c=cfg.max_len_char,
            use_bert=use_bert
        )

        dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True
        )

        test_loop = tqdm(dataloader, desc='[Test by year]', bar_format='{l_bar}{r_bar}', leave=False)

        metrics, preds, labels = test_finetuning(
            model, device, test_loop, use_bert
        )

        # ===== 출력 =====
        print(f"\nTesting data for 20{year}")
        print(
            f"Accuracy: {metrics['accuracy']:.4f}, "
            f"Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}, "
            f"F1: {metrics['f1']:.4f}, "
            f"FPR: {metrics['fpr']:.4f}, FNR: {metrics['fnr']:.4f}"
        )

        if args.use_wandb:
            log_dict = {
                f"Year/{year}/Accuracy": metrics["accuracy"],
                f"Year/{year}/Precision": metrics["precision"],
                f"Year/{year}/Recall": metrics["recall"],
                f"Year/{year}/F1": metrics["f1"],
                f"Year/{year}/FPR": metrics["fpr"],
                f"Year/{year}/FNR": metrics["fnr"],
            }
            wandb.log(log_dict)

        acc_all.append(metrics["accuracy"])
        pre_all.append(metrics["precision"])
        rec_all.append(metrics["recall"])
        f1_all.append(metrics["f1"])
        fpr_all.append(metrics["fpr"])
        fnr_all.append(metrics["fnr"])
        year_str.append(f"20{year}")

        global_preds.append(preds)
        global_labels.append(labels)

    # ---- Save year-wise results
    df = pl.DataFrame({
        "Year": year_str,
        "Accuracy": acc_all,
        "Precision": pre_all,
        "Recall": rec_all,
        "F1_Score": f1_all,
        "FPR": fpr_all,
        "FNR": fnr_all
    })

    if args.save:
        save_path = path_figure.joinpath("test_by_year.csv")
        df.write_csv(save_path)

    print("\n===== Year-wise Results =====")
    print(df)

    # ---- Global metrics
    global_preds = torch.cat(global_preds)
    global_labels = torch.cat(global_labels)

    global_metrics = compute_metrics(global_labels, global_preds)
    print("\n===== Overall Metrics =====")
    print(
        f"[Overall] "
        f"Acc={global_metrics['accuracy']:.4f} | "
        f"P={global_metrics['precision']:.4f} | "
        f"R={global_metrics['recall']:.4f} | "
        f"F1={global_metrics['f1']:.4f} | " 
        f"FPR={global_metrics['fpr']:.4f} | "
        f"FNR={global_metrics['fnr']:.4f}"
    )

    if args.use_wandb:
        wandb.log({
            "Global/Accuracy": global_metrics["accuracy"],
            "Global/Precision": global_metrics["precision"],
            "Global/Recall": global_metrics["recall"],
            "Global/F1": global_metrics["f1"],
            "Global/FPR": global_metrics["fpr"],
            "Global/FNR": global_metrics["fnr"],
        })


def test_by_family(cfg, args, model, tokenizer, device, use_bert=False):

    acc_all = []
    pre_all = []
    rec_all = []
    f1_all  = []
    fpr_all = []
    fnr_all = []

    family_list = []

    paths = glob.glob(str(path_dga_scheme.joinpath('*.parquet')))
    for path in paths:
        family_list.append(pathlib.Path(path).stem)

    family_list.sort()

    for family in family_list:
        test_path = path_dga_scheme.joinpath(f"{family}.parquet")
        test_df = pl.read_parquet(test_path)

        test_dataset = FineTuningDataset(
            test_df,
            tokenizer=tokenizer,
            max_len_t=cfg.max_len_subword,
            max_len_c=cfg.max_len_char,
            use_bert=use_bert
        )

        dataloader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True
        )

        test_loop = tqdm(dataloader, desc='[Test by family]', bar_format='{l_bar}{r_bar}', leave=False)

        metrics, _, _ = test_finetuning(
            model, device, test_loop, use_bert
        )

        acc_all.append(metrics["accuracy"])
        pre_all.append(metrics["precision"])
        rec_all.append(metrics["recall"])
        f1_all.append(metrics["f1"])
        fpr_all.append(metrics["fpr"])
        fnr_all.append(metrics["fnr"])

        # ===== 출력 =====
        print(f"\nTesting data for family: {family}")
        print(
            f"Accuracy: {metrics['accuracy']:.4f}, "
            f"Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}, "
            f"F1: {metrics['f1']:.4f}"
        )
        print(
            f"FPR: {metrics['fpr']:.4f}, "
            f"FNR: {metrics['fnr']:.4f}"
        )
        print(
            f"TN: {metrics['tn']}, FP: {metrics['fp']}, "
            f"FN: {metrics['fn']}, TP: {metrics['tp']}"
        )

        if args.use_wandb:
            wandb.log({
                f"Family/{family}/Accuracy": metrics["accuracy"],
                f"Family/{family}/Precision": metrics["precision"],
                f"Family/{family}/Recall": metrics["recall"],
                f"Family/{family}/F1": metrics["f1"],
                f"Family/{family}/FPR": metrics["fpr"],
                f"Family/{family}/FNR": metrics["fnr"],
            })

    # ===== 결과 정리 =====
    results_df = pl.DataFrame({
        "Family": family_list,
        "Accuracy": acc_all,
        "Precision": pre_all,
        "Recall": rec_all,
        "F1_Score": f1_all,
        "FPR": fpr_all,
        "FNR": fnr_all
    })

    print("\n===== Family-wise Results =====")
    print(results_df)

    if args.save:
        save_path = path_figure.joinpath("test_by_family.csv")
        results_df.write_csv(save_path)


def main() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test_type", type=str, choices=["year", "family"], default="year", help="Test type")
    parser.add_argument("--clf_norm", type=str, default='cls', choices=['cls', 'pool'])
    parser.add_argument("--save", type=bool, default=False, help="Save results")
    parser.add_argument("--project_name", type=str, default="proposal", help="Wandb project name")
    parser.add_argument("--run_name", type=str, default="run", help="Wandb run name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--use_bert", default=False, help="Use BERT backbone")
    parser.add_argument("--tokenizer_min_freq", type=int, default=2, help="Tokenizer min frequency")
    parser.add_argument("--tokenizer_vocab_size", type=int, default=32393, help="Tokenizer vocab size")

    args = parser.parse_args()
    args.use_wandb = not args.no_wandb

    cfg = PretrainConfig(
        min_freq_subword=args.tokenizer_min_freq,
        vocab_size_subword=args.tokenizer_vocab_size
    )

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(path_tokenizer.joinpath(f"tokenizer-{cfg.min_freq_subword}-{cfg.vocab_size_subword}-both-tld.json")))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pt_model_c = PretrainedModel(
        vocab_size=cfg.vocab_size_char,
        d_model=cfg.d_model,
        n_heads=cfg.nhead,
        dim_feedforward=cfg.dim_feedforward,
        num_layers=cfg.num_layers,
        max_len=cfg.max_len_char
    )

    pt_model_t = PretrainedModel(
        vocab_size=cfg.vocab_size_subword,
        d_model=cfg.d_model,
        n_heads=cfg.nhead,
        dim_feedforward=cfg.dim_feedforward,
        num_layers=cfg.num_layers,
        max_len=cfg.max_len_subword
    )

    model = FineTuningModel(pt_model_t, pt_model_c, clf_norm=args.clf_norm, use_bert=args.use_bert).to(device)

    state = torch.load(path_model.joinpath(args.model_path), map_location=device)
    model.load_state_dict(state, strict=False)
    
    if args.use_wandb:
        wandb.init(project=args.project_name, name=args.run_name, config=vars(cfg), tags = ['valid'])

    if args.test_type == "year" :
        test_by_year(cfg, args, model, tokenizer, device, use_bert=args.use_bert)
    elif args.test_type == "family" :
        test_by_family(cfg, args, model, tokenizer, device, use_bert=args.use_bert)

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__" :
    main()