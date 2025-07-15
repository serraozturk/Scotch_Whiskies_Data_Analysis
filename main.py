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

if __name__ == '__main__':
    main()
