import pandas as pd
import glob
import os
from tqdm import tqdm
from dataset_processor_tld import wrap_tld

# 추출기 객체 생성 (캐싱을 통해 성능 최적화)
extract = tldextract.TLDExtract(include_psl_private_domains=True)

def wrap_tld_previous(domain):
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

def debug_tld_processing(patterns, type, year=None):
    """
    기존 merging 로직을 참고하여, 전처리 후 '['로 시작하게 되는 
    이상 케이스의 원본과 결과물을 비교 출력합니다.
    """
    all_paths = []
    for p in patterns:
        all_paths.extend(glob.glob(p))

    results = []
    return_df = []

    for file_path in tqdm(all_paths, desc=f"Debugging {type}"):
        file_name = os.path.basename(file_path)
        try:
            # 1. 기존 merging 함수와 동일한 방식으로 파일 로드
            if type == "benign":
                df = pd.read_csv(file_path, header=None)
                # benign_data 기준: 1번 인덱스 컬럼이 domain
                df = df.rename(columns={df.columns[1]: "original_domain"})
            elif type == "dga":
                df = pd.read_csv(file_path)
                # dga_data 기준: 0번 인덱스 컬럼이 domain
                df = df.rename(columns={df.columns[0]: "original_domain"})
                
                # DGA 전용 날짜 필터링 적용 (기존 로직 유지)
                if year:
                    df['valid_from'] = pd.to_datetime(df['valid_from'])
                    df = df[(f'20{year}-01-01' <= df['valid_from']) & 
                            (df['valid_from'] < f'20{int(year)+1}-01-01')]

            # 2. 분석에 필요한 컬럼만 추출 및 정리
            debug_df = df[["original_domain"]].copy()
            debug_df['original_domain'] = debug_df['original_domain'].astype(str).str.lower()

            # 3. wrap_tld 적용 (처리 후 결과를 별도 컬럼에 저장)
            debug_df['processed_domain'] = debug_df['original_domain'].apply(wrap_tld_previous)
            debug_df['processed_domain2'] = debug_df['original_domain'].apply(wrap_tld)

            # 4. 필터링: 처리 후 결과가 '['로 시작하는 경우 (prefix가 사라진 경우)
            error_cases = debug_df[debug_df['processed_domain'].str.startswith('[', na=False)].copy()

            # 5. 원본 도메인이 . 으로 split했을 때 3개 요소가 나오는 경우만 필터링
            three_part_domains = error_cases[error_cases['original_domain'].str.split('.').str.len() == 3].copy()
            
            if not error_cases.empty:
                error_cases['source_file'] = file_name
                results.append(error_cases)

            if not three_part_domains.empty:
                three_part_domains['source_file'] = file_name
                return_df.append(three_part_domains)

        except Exception as e:
            print(f"{file_name} 로드 실패: {e}")

    # 5. 결과 합치기 및 중복 제거
    if results:
        final_debug_df = pd.concat(results, ignore_index=True)
        final_debug_df = final_debug_df.drop_duplicates(subset=['original_domain'])

    if return_df:
        final_return_df = pd.concat(return_df, ignore_index=True)
        final_return_df = final_return_df.drop_duplicates(subset=['original_domain'])

        return final_debug_df, final_return_df
    else:
        return pd.DataFrame(columns=['original_domain', 'processed_domain', 'source_file']), pd.DataFrame(columns=['original_domain', 'processed_domain', 'source_file'])

if __name__ == "__main__":
    year = '23'  # 테스트하고 싶은 연도
    data_dir = '/home/chlee/codes/Deep-learning-based-DGA_origin/dataset'
    
    # Benign 데이터 디버깅 실행
    benign_patterns = [os.path.join(data_dir, f'tranco_API/20{year}/tranco*.csv')]
    bad_benign, three_part_benign = debug_tld_processing(benign_patterns, 'benign')

    # dga
    dga_patterns = [os.path.join(data_dir, f'2024-11-28-dgarchive_full/*.csv')]
    bad_dga, three_part_dga = debug_tld_processing(dga_patterns, 'dga')
    
    print(f"\n[Benign] 이상 케이스 발견: {len(bad_benign)}건")
    print(bad_benign)
    print(f"\n[Benign] 3부분 도메인 발견: {len(three_part_benign)}건")
    print(three_part_benign)

    print(f"\n[DGA] 이상 케이스 발견: {len(bad_dga)}건")
    print(bad_dga)
    print(f"\n[DGA] 3부분 도메인 발견: {len(three_part_dga)}건")
    print(three_part_dga)

    save_dir = '/home/chlee/codes/Deep-learning-based-DGA/dataset'
    bad_benign.to_csv(os.path.join(save_dir, f'test/tranco_20{year}-bad_benign.csv'), index=False)
    three_part_benign.to_csv(os.path.join(save_dir, f'test/tranco_20{year}-three_part_benign.csv'), index=False)
    bad_dga.to_csv(os.path.join(save_dir, f'test/dgarchive_20{year}-bad_dga.csv'), index=False)
    three_part_dga.to_csv(os.path.join(save_dir, f'test/dgarchive_20{year}-three_part_dga.csv'), index=False)


    # include_pls_private_domains=Flase 옵션일 때 기업이름 잘 인식하는지 테스트
    ext_f = extract_false("blog.naver.com")
    result = get_processed_result(ext_f)
    print(result)

    # 새 wrap_tld함수(wrap_tld로 이름 최종 수정함)의 결과가 이와 동일한지 (즉 include_pls 옵션을 잠시 Flase로 끄는 처리로 들어간 값을 최종 return하는지 체크)
    print(wrap_tld("blog.naver.com"))

    