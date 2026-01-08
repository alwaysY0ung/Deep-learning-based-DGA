import pandas as pd
import polars as pl
import glob
import os
import re
from path import path_dir_root
from config import DatasetConfig
import tqdm
import tldextract
import sys

# 추출기 객체 생성 (캐싱을 통해 성능 최적화)
extract = tldextract.TLDExtract(include_psl_private_domains=True)

def wrap_tld(domain):
    """
    tldextract로 Suffix를 분리한 후, 
    Suffix 내의 마침표를 기준으로 각각 대괄호 처리를 합니다.
    예: naver.co.kr -> naver[.co][.kr]
        sub.example.com -> sub.example[.com]
    """
    if not domain:
        return domain
        
    try:
        ext = extract(domain)
        
        # 1. Suffix(TLD) 가공: "co.kr" -> "[.co][.kr]"
        if ext.suffix:
            # 점으로 나누고 각각 [.부분] 형태로 변환
            wrapped_suffix = "".join([f"[.{part}]" for part in ext.suffix.split('.')])
        else:
            wrapped_suffix = ""

        # 2. 앞부분(Subdomain + Domain) 결합
        main_parts = []
        if ext.subdomain:
            main_parts.append(ext.subdomain)
        if ext.domain:
            main_parts.append(ext.domain)
            
        prefix = ".".join(main_parts)
        
        return f"{prefix}{wrapped_suffix}"
    
    except Exception:
        # 분석 실패 시 원본 혹은 기본 처리 결과 반환
        return domain

def df_prep(df, type):
    if type=="benign":
        df = df.rename(columns={df.columns[1]: "domain"})
        df['label'] = 0
    elif type=="dga":
        df = df.rename(columns={df.columns[0]: "domain"})
        df['label'] = 1
    df = df[["domain", "label"]].copy()
    df['domain'] = df['domain'].str.lower()
    df['domain'] = df['domain'].apply(wrap_tld)
    return df

def merging(patterns, output_file, year, type):
    all_paths = []
    for p in patterns:
        all_paths.extend(glob.glob(p))

    print(f"발견된 경로 목록 (파일 또는 디렉토리): {all_paths}")

    # 최종 결과물을 저장할 빈 데이터프레임 초기화
    df_list = []
    raws = 0

    for file_path in tqdm.tqdm(all_paths, bar_format='{l_bar}{r_bar}'):
        file_name = os.path.basename(file_path)
        try:
            # 1. 파일 읽기 및 전처리
            if type=="benign":
                current_df = pd.read_csv(file_path, header=None)
            elif type=="dga":
                current_df = pd.read_csv(file_path)
                current_df['valid_from'] = pd.to_datetime(current_df['valid_from'])        # 문자열 → datetime 변환
                current_df = current_df[(f'20{year}-01-01' <= current_df['valid_from']) & (current_df['valid_from'] < f'20{int(year)+1}-01-01')]  
            
            current_df = df_prep(current_df, type)
            raws += len(current_df)
            df_list.append(current_df)

            print(f"처리 중: {file_name} | 현재 누적 도메인 수 (중복 허용): {raws}")

        except Exception as e:
            print(f"{file_name} 처리 실패: {e}")

    final_df = pd.concat(df_list, ignore_index=True)
    final_df = final_df.drop_duplicates(subset='domain')
    final_df.to_parquet(output_file, index=False)
    return final_df

def benign_data(year):
    data_dir = '/home/chlee/codes/Deep-learning-based-DGA_origin/dataset'
    output_file = path_dir_root.joinpath(f'dataset/tld/T{year}_benign_tld.parquet')

    patterns = [
        os.path.join(data_dir, f'tranco_API/20{year}/tranco*.csv'),
        os.path.join(data_dir, f'alexa_croll/20{year}*.csv')
    ]

    df = merging(patterns, output_file, year, 'benign')
    print(f"연도 {year} 저장 완료: {output_file} (총 {len(df)}행)")


def dga_data(year):
    data_dir = '/home/chlee/codes/Deep-learning-based-DGA_origin/dataset'
    output_file = path_dir_root.joinpath(f'dataset/tld/T{year}_dga_tld.parquet')

    patterns = [
        os.path.join(data_dir, f'2024-11-28-dgarchive_full/*.csv')]

    df =merging(patterns, output_file, year, 'dga')
    print(f"연도 {year} 저장 완료: {output_file} (총 {len(df)}행)")


def split_parquet_file(file_path, train_size, val_size):
    """
    지정된 Parquet 파일을 로드하고 셔플한 뒤, train, val, test 세트로 분할하여 저장합니다.

    Args:
        file_path (str): 원본 Parquet 파일 경로
        train_size (int): Train 세트의 행 개수
        val_size (int): Validation 세트의 행 개수"""

    # 2. Polars로 파일 로드
    df = pl.read_parquet(file_path)
    total_rows = df.height
    print(f"파일 로드 완료. 총 행 개수: {total_rows}")

    # 3. 데이터 셔플 (전체 데이터를 랜덤하게 섞음)
    # fraction=1.0과 shuffle=True를 사용하면 전체 DataFrame을 셔플
    print("데이터 셔플 중...")
    shuffled_df = df.sample(fraction=1.0, shuffle=True)

    # 4. 데이터 분할 (Slicing)
    # slice(offset, length)
    print("데이터 분할 중...")
    test_start_index = train_size + val_size

    train_df = shuffled_df.slice(0, train_size)
    val_df = shuffled_df.slice(train_size, val_size)
    # slice(offset)은 해당 offset부터 끝까지 모든 데이터를 의미
    test_df = shuffled_df.slice(test_start_index) 

    print(f"분할 완료:")
    print(f"  Train: {train_df.height} 개 (요청: {train_size})")
    print(f"  Val:   {val_df.height} 개 (요청: {val_size})")
    print(f"  Test:  {test_df.height} 개 (나머지)")
    
    # 원본 개수와 분할된 개수의 합이 일치하는지 확인
    if train_df.height + val_df.height + test_df.height != total_rows:
        print(f"[경고] 분할된 행의 총합({train_df.height + val_df.height + test_df.height})이 "
                f"원본({total_rows})과 일치하지 않음!", file=sys.stderr)
    
    # 5. 파일명 설정 및 저장
    base_name, _ = os.path.splitext(file_path)
    
    output_files = {
        f"{base_name}_train.parquet": train_df,
        f"{base_name}_val.parquet": val_df,
        f"{base_name}_test.parquet": test_df,
    }

    for path, data in output_files.items():
        print(f"저장 중: {path}")
        data.write_parquet(path)

if __name__ == "__main__":
    for y in ['17','18','19','20','21','22','23','24','25']:
            # Benign 처리
            benign = path_dir_root.joinpath(f'dataset/tld/T{y}_benign_tld.parquet')
            if not benign.exists():
                benign_data(y)
            
            train_b = path_dir_root.joinpath(f'dataset/tld/T{y}_benign_train_tld.parquet')
            if benign.exists() and not train_b.exists():
                split_parquet_file(str(benign), DatasetConfig.train_size, DatasetConfig.val_size)

            # DGA 처리 (동일한 로직)
            dga = path_dir_root.joinpath(f'dataset/tld/T{y}_dga_tld.parquet')
            if not dga.exists():
                dga_data(y)
                
            train_d = path_dir_root.joinpath(f'dataset/tld/T{y}_dga_train_tld.parquet')
            if dga.exists() and not train_d.exists():
                split_parquet_file(str(dga), DatasetConfig.train_size, DatasetConfig.val_size)