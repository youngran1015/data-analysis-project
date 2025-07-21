import os
import re
import pandas as pd
from pathlib import Path

# 인코딩 순서 (CP949를 먼저 시도)
ENCODINGS = ['cp949', 'euc-kr', 'utf-8', 'iso-8859-1']
RAW_DATA_DIR = 'raw_data'
OUTPUT_DIR = 'data'

# Ensure output directory and subfolders exist
SUBFOLDERS = ['mma', 'kdca', 'dapa']
for sub in SUBFOLDERS:
    os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

def clean_column_names(columns):
    cleaned = []
    for col in columns:
        col = str(col).strip()
        # 한글 컬럼명은 그대로 유지
        col = re.sub(r'[^\w\s가-힣]', '', col)  # 한글 문자 포함
        col = re.sub(r'\s+', '_', col)      # 공백을 언더스코어로
        cleaned.append(col)
    return cleaned

def try_read_csv(filepath):
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            return df, enc
        except Exception:
            continue
    return None, None

def remove_broken_rows(df):
    # Remove rows with all NaN or with wrong number of columns
    df = df.dropna(how='all')
    if hasattr(df, 'columns'):
        expected_cols = len(df.columns)
        df = df[df.apply(lambda row: len(row) == expected_cols, axis=1)]
    return df

def generate_english_filename(df, original_path):
    # Use first 2-3 column names and a hash of the original filename
    base = Path(original_path).stem
    cols = list(df.columns)[:3]
    cols_part = '_'.join([re.sub(r'[^a-zA-Z0-9]', '', c) for c in cols])
    base_part = re.sub(r'[^a-zA-Z0-9]', '', base)
    filename = f"{cols_part}_{base_part}.csv".lower()
    return filename

def process_csv_file(filepath, subfolder):
    df, enc = try_read_csv(filepath)
    if df is None:
        print(f"Failed to read {filepath} with tried encodings.")
        return
    
    print(f"Processing {filepath} with encoding: {enc}")
    df.columns = clean_column_names(df.columns)
    df = remove_broken_rows(df)
    
    # 원본 파일명 사용
    original_filename = Path(filepath).name
    output_path = os.path.join(OUTPUT_DIR, subfolder, original_filename)
    
    # UTF-8로 저장
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved to: {output_path}")

def main():
    for sub in SUBFOLDERS:
        sub_path = os.path.join(RAW_DATA_DIR, sub)
        for root, dirs, files in os.walk(sub_path):
            for file in files:
                if file.lower().endswith('.csv'):
                    filepath = os.path.join(root, file)
                    process_csv_file(filepath, sub)

if __name__ == '__main__':
    main() 