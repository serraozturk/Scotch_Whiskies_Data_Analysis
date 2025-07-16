import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


def main():
    # 1) Loaded and cleaned data
    df = pd.read_csv("Scotch_Whiskies_Data.csv", header=[0,1])
    # Flatten columns: group (header0) + feature (header1)
    cols = []
    for g1, g2 in df.columns:
        g1 = (g1 or '').strip().lower()
        feat = g2.strip().lower().replace('.', '').replace(' ', '_')
        cols.append(f"{g1}_{feat}" if g1 else feat)
    df.columns = cols

    # 2) Remove unnecessary unnamed columns and rename critical columns -- important
    rename_map = {
        'unnamed: 0_level_0_name': 'name',
        'unnamed: 69_level_0_age': 'age',
        'unnamed: 70_level_0_score': 'score',
        'unnamed: 72_level_0_region': 'region',
        'unnamed: 73_level_0_district': 'district'
    }
    df = df.rename(columns=rename_map)
    df = df.drop(columns=[c for c in df.columns if c.endswith('_%') or 'unnamed: 71' in c], errors='ignore')

    # Cleaning done: drop duplicates
    df = df.drop_duplicates().dropna(subset=['name', 'age', 'score'])

    # 3) Type conversions and region labels
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    if 'region' in df.columns:
        # to avoid duplicate labels like 'low' vs 'Low'
        df['region'] = df['region'].astype(str).str.strip().str.title().astype('category')
    if 'district' in df.columns:
        df['district'] = df['district'].astype(str).str.strip().str.title().astype('category')

    # 4) Feature groups (dynamically) based on csv file
    color_feats  = [c for c in df.columns if c.startswith('color_')]
    nose_feats   = [c for c in df.columns if c.startswith('nose_')]
    body_feats   = [c for c in df.columns if c.startswith('body_')]
    pal_feats    = [c for c in df.columns if c.startswith('pal_')]
    fin_feats    = [c for c in df.columns if c.startswith('fin_')]
    finish_feats = pal_feats + fin_feats
    tat_feats    = [c for c in nose_feats if any(x in c for x in ['fruit','sweet','dry','sherry'])]
    aroma_feats  = [c for c in nose_feats if any(x in c for x in ['peat','smoke','sea','grass','spicy'])]



if __name__ == '__main__':
    main()
