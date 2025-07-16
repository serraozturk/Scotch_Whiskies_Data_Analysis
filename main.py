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
        'unnamed: 71_level_0_%': 'percent',
        'unnamed: 72_level_0_region': 'region',
        'unnamed: 73_level_0_district': 'district'
    }
    df = df.rename(columns=rename_map)
    df = df.drop(columns=[c for c in df.columns if 'unnamed: 71' in c], errors='ignore')

    # Cleaning done: drop duplicates
    df = df.drop_duplicates().dropna(subset=['name', 'age', 'score'])

    # 3) Type conversions and region labels
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    if 'region' in df.columns:
        # clean up stray spaces/case
        df['region'] = df['region'].astype(str) \
            .str.strip() \
            .str.title() \
            .astype('category')

    # 4) Feature groups (dynamically) based on csv file
    color_feats  = [c for c in df.columns if c.startswith('color_')]
    nose_feats   = [c for c in df.columns if c.startswith('nose_')]
    body_feats   = [c for c in df.columns if c.startswith('body_')]
    pal_feats    = [c for c in df.columns if c.startswith('pal_')]
    fin_feats    = [c for c in df.columns if c.startswith('fin_')]
    finish_feats = pal_feats + fin_feats
    tat_feats    = [c for c in nose_feats if any(x in c for x in ['fruit','sweet','dry','sherry'])]
    aroma_feats  = [c for c in nose_feats if any(x in c for x in ['peat','smoke','sea','grass','spicy'])]

  # 5) Dominant color category
    if color_feats:
        df['color'] = df[color_feats].idxmax(axis=1).str.replace('color_', '').astype('category')

    # 6) Min–max normalize all features
    all_feats = tat_feats + aroma_feats + finish_feats + body_feats + color_feats
    X = df[all_feats].astype(float)
    Xn = (X - X.min()) / (X.max() - X.min())

    # 7) Calculate category scores
    df['score_tat']    = Xn[tat_feats].mean(axis=1)    if tat_feats else 0
    df['score_aroma']  = Xn[aroma_feats].mean(axis=1)  if aroma_feats else 0
    df['score_finish'] = Xn[finish_feats].mean(axis=1) if finish_feats else 0
    df['score_body']   = Xn[body_feats].mean(axis=1)   if body_feats else 0
    df['score_color']  = Xn[color_feats].mean(axis=1)  if color_feats else 0


if __name__ == '__main__':
    main()
