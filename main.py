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



if __name__ == '__main__':
    main()
