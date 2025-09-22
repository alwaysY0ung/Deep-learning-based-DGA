import pandas as pd
import tqdm

"""
dataset/에 있는 원본 csv 파일들을 불러와서 전처리 후 cache/에 parquet 파일로 저장
아예 raw로부터 domain, type, label 컬럼을 가진 데이터프레임을 만드는 작업
"""

# load dataset: benign
df_benign = pd.read_csv('./dataset/alexa_top_1m_170320.csv', index_col=0, names=['domain'])
df_benign['type'] = 'normal.alexa'
df_benign['label'] = 0

# load dataset: dga
df_dga_org = pd.read_csv('./dataset/test_dga.csv') # https://huggingface.co/datasets/harpomaxx/dga-detection
df_dga = df_dga_org[~df_dga_org['label'].str.contains('normal')] # benign 도메인이 섞여있음. 'label'값이 normal.alexa
df_dga = df_dga.drop(columns=['id', 'date'])
df_dga = df_dga.rename(columns={"domain" : "domain", "label" : "type"})
df_dga['label'] = 1
df_dga = df_dga.reset_index(drop=True) # drop=True가 없으면 기존 index가 'index'라는 이름의 새로운 컬럼으로 추가됨

print(df_benign.head())
print(df_benign.shape)

print(df_dga.head())
print(df_dga.shape)

df_benign.to_parquet('./cache/alexa.parquet', index=False) # pd가 아니고 pyarrow가 설치되어있어야함.
df_dga.to_parquet('./cache/harpomaxx.parquet', index=False) 


print(df_benign.info())
print(df_dga.info())