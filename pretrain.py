import argparse
import polars as pl
pl.Config.set_engine_affinity(engine="streaming")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocessing import SubTaskDataset, SpecialIDs
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast
from model import PretrainedModel, PretrainMamba
from tqdm import tqdm
import datetime
import wandb
from utility.dataset import get_train_set_tld, get_val_set_tld
from utility.config import PretrainConfig
from utility.path import path_model, path_tokenizer
from make_tokenizer_tld import train


def log_artifact(run, path, name, type_="model"):
    if run is not None:
        artifact = wandb.Artifact(name=name, type=type_)
        artifact.add_file(path)
        run.log_artifact(artifact)


def save_checkpoint(model, optimizer, global_step, best_loss, 
                    interval_loss_sum_total, interval_loss_sum_mtp, interval_loss_sum_tpp, interval_loss_sum_tov, save_path) :
    torch.save({
        'global_step': global_step,
        'best_loss': best_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'interval_loss_sum_total': interval_loss_sum_total,
        'interval_loss_sum_mtp': interval_loss_sum_mtp,
        'interval_loss_sum_tpp': interval_loss_sum_tpp,
        'interval_loss_sum_tov': interval_loss_sum_tov,
    }, save_path)


def train_char(cfg, args) :
    now_date = datetime.datetime.now().strftime('%m%d_%H%M')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run = None
    if args.use_wandb:
        if args.run_id is not None :
            run = wandb.init(project=args.project_name, id=args.run_id, resume='allow')
        run = wandb.init(project=args.project_name, name=args.run_name, config=vars(cfg), tags=['valid'])

    tokenizer_path = path_tokenizer.joinpath((f"tokenizer-{cfg.min_freq_subword}-{cfg.vocab_size_subword}-both-tld.json"))
    if not tokenizer_path.exists():
        _, paths = get_train_set_tld()
        train(file_paths=paths,
            text_col="domain",
            vocab_size=cfg.vocab_size_subword,
            min_freq=cfg.min_freq_subword,
            use_bert_pretokenizer=True,
            save_path=tokenizer_path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    print(f"Loaded Tokenizer Vocab Size: {tokenizer.vocab_size}")
    assert tokenizer.vocab_size == cfg.vocab_size_subword, "Tokenizer vocab size does not match!"

    train_df, _ = get_train_set_tld() # .sample(frac=0.01) for debugging
    val_df = get_val_set_tld() # .sample(frac=0.01) for debugging

    train_dataset = SubTaskDataset(
        train_df,
        max_len=cfg.max_len_char,
        tokenizer=tokenizer,
        mask_ratio=cfg.mask_ratio,
        type = 'char'
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    val_dataset = SubTaskDataset(
        val_df,
        max_len=cfg.max_len_char,
        tokenizer=tokenizer,
        mask_ratio=cfg.mask_ratio,
        type = 'char'
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    if args.type == "transformer" :
        model = PretrainedModel(
            vocab_size=cfg.vocab_size_char,
            d_model=cfg.d_model,
            n_heads=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            num_layers=cfg.num_layers,
            max_len=cfg.max_len_char,
            tov_norm=cfg.tov_norm,
        ).to(device)
    elif args.type == "mamba" :
        model = PretrainMamba(
            vocab_size=cfg.vocab_size_char,
            d_model=cfg.d_model,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            padding_idx=cfg.padding_idx,
            mamba_bidirectional=args.bidirectional,
            tov_norm=cfg.tov_norm,
        ).to(device)

    ce = nn.CrossEntropyLoss(ignore_index=cfg.ignore_index)
    bce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    train_loop = tqdm(total=args.total_steps, desc="[Train]", bar_format='{l_bar}{r_bar}')
    best_loss = float('inf')

    if args.checkpoint_path is not None :
        checkpoint = torch.load(path_model.joinpath(args.checkpoint_path))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint['global_step']
        interval_loss_sum_total = checkpoint['interval_loss_sum_total']
        interval_loss_sum_mtp = checkpoint['interval_loss_sum_mtp']
        interval_loss_sum_tpp = checkpoint['interval_loss_sum_tpp']
        interval_loss_sum_tov = checkpoint['interval_loss_sum_tov']
        train_loop.update(global_step)
    else :
        global_step = 0
        interval_loss_sum_total = 0 
        interval_loss_sum_mtp = 0
        interval_loss_sum_tpp = 0
        interval_loss_sum_tov = 0

    model.train()

    while global_step < args.total_steps :

        total_mtp_loss = 0
        total_tpp_loss = 0
        total_tov_loss = 0

        try:
            for X_mtp, Y_mtp, X_tpp, Y_tpp, X_tov, Y_tov in train_dataloader :
                if global_step >= args.total_steps :
                    break

                X_mtp, Y_mtp = X_mtp.to(device), Y_mtp.to(device)
                X_tpp, Y_tpp = X_tpp.to(device), Y_tpp.to(device)
                X_tov, Y_tov = X_tov.to(device), Y_tov.to(device)

                # --- T1: MTP Loss 계산 ---
                logits_mtp = model(X_mtp, task_type='MTP')
                loss_mtp = ce(
                    logits_mtp.view(-1, logits_mtp.size(-1)), # (B*L, V)
                    Y_mtp.view(-1)                            # (B*L)
                )

                # --- T2: TPP Loss 계산 ---
                logits_tpp = model(X_tpp, task_type='TPP')
                loss_tpp = ce(
                    logits_tpp.view(-1, logits_tpp.size(-1)), # (B*L, L)
                    Y_tpp.view(-1)                             # (B*L)
                )

                # --- T3: TOV Loss 계산 ---
                logits_tov = model(X_tov, task_type='TOV')
                loss_tov = bce(logits_tov, Y_tov) # Logits: (B x 2), Labels: (B)

                L_total = loss_mtp + loss_tpp + loss_tov

                optimizer.zero_grad()
                L_total.backward()
                optimizer.step()

                total_mtp_loss += loss_mtp.item()
                total_tpp_loss += loss_tpp.item()
                total_tov_loss += loss_tov.item()

                interval_loss_sum_total += L_total.item()
                interval_loss_sum_mtp += loss_mtp.item()
                interval_loss_sum_tpp += loss_tpp.item()
                interval_loss_sum_tov += loss_tov.item()

                train_loop.update(1)
                global_step += 1
                
                # 인터벌 로깅
                if global_step % args.log_interval == 0:
                    avg_total_interval = interval_loss_sum_total / args.log_interval
                    avg_mtp_interval = interval_loss_sum_mtp / args.log_interval
                    avg_tpp_interval = interval_loss_sum_tpp / args.log_interval
                    avg_tov_interval = interval_loss_sum_tov / args.log_interval
                    
                    if args.use_wandb:
                        wandb.log({
                            "step/interval_total_loss": avg_total_interval,
                            "step/interval_mtp_loss": avg_mtp_interval,
                            "step/interval_tpp_loss": avg_tpp_interval,
                            "step/interval_tov_loss": avg_tov_interval,
                        }, step=global_step//args.log_interval -1)

                    train_loop.write(f"[Step {global_step} Interval Log]: Train Loss: {avg_total_interval:.4f}")

                    interval_loss_sum_total = 0 
                    interval_loss_sum_mtp = 0
                    interval_loss_sum_tpp = 0
                    interval_loss_sum_tov = 0

                if global_step % args.val_check_interval == 0:
                    val_loss = validate(model, val_dataloader, device, global_step, cfg, args)
                    train_loop.write(f"[char] step {global_step} val_loss={val_loss:.4f}")

                    if val_loss < best_loss:
                        best_loss = val_loss
                        save_path = path_model.joinpath(f"{now_date}_{cfg.save_path.replace('.pt', '')}_step_{global_step}.pt")
                        torch.save(model.state_dict(), save_path)
                        if args.use_wandb:
                            pass
                            # log_artifact(run, save_path, f"{args.mode}_{now_date}") # wandb artifact 저장 필요 시

                current_step = train_loop.n
                with torch.no_grad() :
                    avg_mtp = total_mtp_loss / current_step
                    avg_tpp = total_tpp_loss / current_step
                    avg_tov = total_tov_loss / current_step
                    avg_total = (total_mtp_loss + total_tpp_loss + total_tov_loss) / current_step

                train_loop.set_postfix(dict(avg_total=f'{avg_total:.4f}',
                                        avg_mtp=f'{avg_mtp:.4f}',
                                        avg_tpp=f'{avg_tpp:.4f}',
                                        avg_tov=f'{avg_tov:.4f}'), refresh=False)
        except (KeyboardInterrupt, Exception) as e:
            print(f"\n{type(e).__name__} — saving checkpoint")
            save_path = path_model.joinpath(f"checkpoint-{global_step}.pt")
            save_checkpoint(model, optimizer, global_step, best_loss, interval_loss_sum_total, interval_loss_sum_mtp, interval_loss_sum_tpp, interval_loss_sum_tov, save_path)
            print(f"\nmodel save at {global_step}")
            break

    if args.use_wandb :
        wandb.finish()


def train_subword(cfg, args) :
    now_date = datetime.datetime.now().strftime('%m%d_%H%M')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run = None
    if args.use_wandb:
        if args.run_id is not None :
            run = wandb.init(project=args.project_name, id=args.run_id, resume='allow')
        run = wandb.init(project=args.project_name, name=args.run_name, config=vars(cfg), tags=['valid'])

    tokenizer_path = path_tokenizer.joinpath((f"tokenizer-{cfg.min_freq_subword}-{cfg.vocab_size_subword}-both-tld.json"))
    if not tokenizer_path.exists():
        _, paths = get_train_set_tld()
        train(file_paths=paths,
            text_col="domain",
            vocab_size=cfg.vocab_size_subword,
            min_freq=cfg.min_freq_subword,
            use_bert_pretokenizer=True,
            save_path=tokenizer_path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    print(f"Loaded Tokenizer Vocab Size: {tokenizer.vocab_size}")
    assert tokenizer.vocab_size == cfg.vocab_size_subword, "Tokenizer vocab size does not match!"

    train_df, _ = get_train_set_tld()
    val_df = get_val_set_tld()

    train_dataset = SubTaskDataset(
        train_df,
        max_len=cfg.max_len_subword,
        tokenizer=tokenizer,
        mask_ratio=cfg.mask_ratio,
        type = 'subword'
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    val_dataset = SubTaskDataset(
        val_df,
        max_len=cfg.max_len_subword,
        tokenizer=tokenizer,
        mask_ratio=cfg.mask_ratio,
        type = 'subword'
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    if args.type == "transformer" :
        model = PretrainedModel(
            vocab_size=cfg.vocab_size_subword,
            d_model=cfg.d_model,
            n_heads=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            num_layers=cfg.num_layers,
            max_len=cfg.max_len_subword,
            tov_norm=cfg.tov_norm,
        ).to(device)
    elif args.type == "mamba" :
        model = PretrainMamba(
            vocab_size=cfg.vocab_size_subword,
            d_model=cfg.d_model,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            padding_idx=cfg.padding_idx,
            mamba_bidirectional=args.bidirectional,
            tov_norm=cfg.tov_norm,
        ).to(device)

    ce = nn.CrossEntropyLoss(ignore_index=cfg.ignore_index)
    bce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    train_loop = tqdm(total=args.total_steps, desc="[Train]", bar_format='{l_bar}{r_bar}')
    best_loss = float('inf')

    if args.checkpoint_path is not None :
        checkpoint = torch.load(path_model.joinpath(args.checkpoint_path))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint['global_step']
        interval_loss_sum_total = checkpoint['interval_loss_sum_total']
        interval_loss_sum_mtp = checkpoint['interval_loss_sum_mtp']
        interval_loss_sum_tpp = checkpoint['interval_loss_sum_tpp']
        interval_loss_sum_tov = checkpoint['interval_loss_sum_tov']
        train_loop.update(global_step)
    else :
        global_step = 0
        interval_loss_sum_total = 0 
        interval_loss_sum_mtp = 0
        interval_loss_sum_tpp = 0
        interval_loss_sum_tov = 0

    model.train()

    while global_step < args.total_steps :

        total_mtp_loss = 0
        total_tpp_loss = 0
        total_tov_loss = 0
        
        try:
            for X_mtp, Y_mtp, X_tpp, Y_tpp, X_tov, Y_tov in train_dataloader :
                
                if global_step >= args.total_steps :
                    break

                X_mtp, Y_mtp = X_mtp.to(device), Y_mtp.to(device)
                X_tpp, Y_tpp = X_tpp.to(device), Y_tpp.to(device)
                X_tov, Y_tov = X_tov.to(device), Y_tov.to(device)

                # --- T1: MTP Loss 계산 ---
                logits_mtp = model(X_mtp, task_type='MTP')
                loss_mtp = ce(
                    logits_mtp.view(-1, logits_mtp.size(-1)), # (B*L, V)
                    Y_mtp.view(-1)                            # (B*L)
                )

                # --- T2: TPP Loss 계산 ---
                logits_tpp = model(X_tpp, task_type='TPP')
                loss_tpp = ce(
                    logits_tpp.view(-1, logits_tpp.size(-1)), # (B*L, L)
                    Y_tpp.view(-1)                             # (B*L)
                )

                # --- T3: TOV Loss 계산 ---
                logits_tov = model(X_tov, task_type='TOV')
                loss_tov = bce(logits_tov, Y_tov) # Logits: (B x 2), Labels: (B)

                L_total = loss_mtp + loss_tpp + loss_tov

                optimizer.zero_grad()
                L_total.backward()
                optimizer.step()

                total_mtp_loss += loss_mtp.item()
                total_tpp_loss += loss_tpp.item()
                total_tov_loss += loss_tov.item()

                interval_loss_sum_total += L_total.item()
                interval_loss_sum_mtp += loss_mtp.item()
                interval_loss_sum_tpp += loss_tpp.item()
                interval_loss_sum_tov += loss_tov.item()

                train_loop.update(1)
                global_step += 1
                
                # 인터벌 로깅
                if global_step % args.log_interval == 0:
                    avg_total_interval = interval_loss_sum_total / args.log_interval
                    avg_mtp_interval = interval_loss_sum_mtp / args.log_interval
                    avg_tpp_interval = interval_loss_sum_tpp / args.log_interval
                    avg_tov_interval = interval_loss_sum_tov / args.log_interval
                    
                    if args.use_wandb:
                        wandb.log({
                            "step/interval_total_loss": avg_total_interval,
                            "step/interval_mtp_loss": avg_mtp_interval,
                            "step/interval_tpp_loss": avg_tpp_interval,
                            "step/interval_tov_loss": avg_tov_interval,
                        }, step=global_step//args.log_interval -1)

                    train_loop.write(f"[Step {global_step} Interval Log]: Train Loss: {avg_total_interval:.4f}")

                    interval_loss_sum_total = 0 
                    interval_loss_sum_mtp = 0
                    interval_loss_sum_tpp = 0
                    interval_loss_sum_tov = 0

                if global_step % args.val_check_interval == 0:
                    val_loss = validate(model, val_dataloader, device, global_step, cfg, args)
                    train_loop.write(f"[subword] step {global_step} val_loss={val_loss:.4f}")

                    if val_loss < best_loss:
                        best_loss = val_loss
                        save_path = path_model.joinpath(f"{now_date}_{cfg.save_path.replace('.pt', '')}_step_{global_step}.pt")
                        torch.save(model.state_dict(), save_path)
                        if args.use_wandb:
                            pass
                            # log_artifact(run, save_path, f"{args.mode}_{now_date}") # wandb artifact 저장 필요 시
                
                current_step = train_loop.n
                with torch.no_grad() :
                    avg_mtp = total_mtp_loss / current_step
                    avg_tpp = total_tpp_loss / current_step
                    avg_tov = total_tov_loss / current_step
                    avg_total = (total_mtp_loss + total_tpp_loss + total_tov_loss) / current_step

                train_loop.set_postfix(dict(avg_total=f'{avg_total:.4f}',
                                        avg_mtp=f'{avg_mtp:.4f}',
                                        avg_tpp=f'{avg_tpp:.4f}',
                                        avg_tov=f'{avg_tov:.4f}'), refresh=False)
        except (KeyboardInterrupt, Exception) as e:
            print(f"\n{type(e).__name__} — saving checkpoint")
            save_path = path_model.joinpath(f"checkpoint-{global_step}.pt")
            save_checkpoint(model, optimizer, global_step, best_loss, interval_loss_sum_total, interval_loss_sum_mtp, interval_loss_sum_tpp, interval_loss_sum_tov, save_path)
            print(f"\nmodel save at {global_step}")
            break

    if args.use_wandb :
        wandb.finish()


def validate(model, dataloader, device, global_step, cfg, args):
    model.eval()
    total_loss = 0
    mtp_loss_total = 0
    tpp_loss_total = 0
    tov_loss_total = 0

    ce = torch.nn.CrossEntropyLoss(ignore_index=cfg.ignore_index)
    bce = torch.nn.CrossEntropyLoss()

    val_loop = tqdm(dataloader, desc="Validation", bar_format='{l_bar}{r_bar}', leave=False)
    
    with torch.no_grad():
        for X_mtp, Y_mtp, X_tpp, Y_tpp, X_tov, Y_tov in val_loop :
            
            X_mtp, Y_mtp = X_mtp.to(device), Y_mtp.to(device)
            X_tpp, Y_tpp = X_tpp.to(device), Y_tpp.to(device)
            X_tov, Y_tov = X_tov.to(device), Y_tov.to(device)

            # --- T1: MTP Loss 계산 ---
            logits_mtp = model(X_mtp, task_type='MTP')
            loss_mtp = ce(
                logits_mtp.view(-1, logits_mtp.size(-1)), # (B*L, V)
                Y_mtp.view(-1)                            # (B*L)
            )

            # --- T2: TPP Loss 계산 ---
            logits_tpp = model(X_tpp, task_type='TPP')
            loss_tpp = ce(
                logits_tpp.view(-1, logits_tpp.size(-1)), # (B*L, L)
                Y_tpp.view(-1)                             # (B*L)
            )

            # --- T3: TOV Loss 계산 ---
            logits_tov = model(X_tov, task_type='TOV')
            loss_tov = bce(logits_tov, Y_tov) # Logits: (B x 2), Labels: (B)

            L_total = loss_mtp + loss_tpp + loss_tov

            total_loss += L_total.item()
            mtp_loss_total += loss_mtp.item()
            tpp_loss_total += loss_tpp.item()
            tov_loss_total += loss_tov.item()

            val_loop.update(1)

    avg_total = total_loss / len(dataloader)
    avg_mtp = mtp_loss_total / len(dataloader)
    avg_tpp = tpp_loss_total / len(dataloader)
    avg_tov = tov_loss_total / len(dataloader)

    if args.use_wandb:
        wandb.log({
            "step/val_total_loss": avg_total,
            "step/val_mtp_loss": avg_mtp,
            "step/val_tpp_loss": avg_tpp,
            "step/val_tov_loss": avg_tov,
        },step=global_step//args.log_interval -1)
    
    model.train()

    return avg_total


def main() :
    parser = argparse.ArgumentParser()
    cfg = PretrainConfig()
    parser.add_argument("--mode", choices=["char", "subword"], required=True,
                        help="Pre-training mode: char or subword")
    parser.add_argument("--type", choices=["transformer", "mamba"], required=True,
                        help="Pre-training type: transformer or mamba")
    parser.add_argument("--bidirectional", default=False, type=bool, help="Use bidirectional mamba if True")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--run_id", type=str, default=None, help="Wandb run id")
    parser.add_argument("--save", type=str, default="pretrained.pt", help="Path to save model state dict")
    parser.add_argument("--total_steps", type=int, default=10000000, help="Total training steps")
    parser.add_argument("--val_check_interval", type=int, default=20000, help="Steps between validation")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--log_interval", type=int, default=1000, help="Steps between logging")
    parser.add_argument("--project_name", type=str, default="dga-pretrain", help="Wandb project name")
    parser.add_argument("--run_name", type=str, default="run", help="Wandb run name")
    parser.add_argument("--tov_norm", type=str, choices=["cls", "pool"], default="cls", help="TOV pooling strategy")
    parser.add_argument("--tokenizer_min_freq", type=int, default=cfg.min_freq_subword, help="Tokenizer min frequency")
    parser.add_argument("--tokenizer_vocab_size", type=int, default=cfg.vocab_size_subword, help="Tokenizer vocab size")

    args = parser.parse_args()
    if args.checkpoint_path is not None and args.run_id is None:
        parser.error("--run_id is required when --checkpoint_path is provided")
    args.use_wandb = not args.no_wandb

    cfg = PretrainConfig(
        save_path=args.save, 
        tov_norm=args.tov_norm,
        min_freq_subword=args.tokenizer_min_freq,
        vocab_size_subword=args.tokenizer_vocab_size
    )

    if args.type == "transformer" and args.bidirectional :
        raise ValueError(f"{'-' * 20}\nBidirectional is not supported for transformer.\n \
            There's only three case for model type: transformer, mamba, mamba-bidirectional.\n{'-' * 20}")

    print(f"--mode: {args.mode}")
    print(f"--bidirectional: {args.bidirectional}")
    print(f"--checkpoint_path: {args.checkpoint_path}")
    print(f"--run_id: {args.run_id}")
    print(f"--save: {args.save}")
    print(f"--total_steps: {args.total_steps}")
    print(f"--val_check_interval: {args.val_check_interval}")
    print(f"--no_wandb: {args.no_wandb}")
    print(f"--log_interval: {args.log_interval}")
    print(f"--project_name: {args.project_name}")
    print(f"--run_name: {args.run_name}")
    print(f"--tov_norm: {args.tov_norm}")
    print(f"--tokenizer_min_freq: {args.tokenizer_min_freq}")
    print(f"--tokenizer_vocab_size: {args.tokenizer_vocab_size}")

    print(f"--save_path: {cfg.save_path}")
    print(f"--tov_norm: {cfg.tov_norm}")
    print(f"--min_freq_subword: {cfg.min_freq_subword}")
    print(f"--vocab_size_subword: {cfg.vocab_size_subword}")

    if args.mode == "char" :
        train_char(cfg, args)
    elif args.mode == "subword" :
        train_subword(cfg, args)

if __name__ == '__main__':
    main()