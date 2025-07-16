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
    all_feats = tat_feats + aroma_feats + finish_feats + body_feats + color_feats + ['age','percent']
    X = df[all_feats].astype(float)
    Xn = (X - X.min()) / (X.max() - X.min())

    # 7) Calculate category scores
    df['score_tat']    = Xn[tat_feats].mean(axis=1)    if tat_feats else 0
    df['score_aroma']  = Xn[aroma_feats].mean(axis=1)  if aroma_feats else 0
    df['score_finish'] = Xn[finish_feats].mean(axis=1) if finish_feats else 0
    df['score_body']   = Xn[body_feats].mean(axis=1)   if body_feats else 0
    df['score_color']  = Xn[color_feats].mean(axis=1)  if color_feats else 0
    df['score_age'] = Xn['age']
    df['score_percent'] = Xn['percent']

    # 8)  similarity
    weights = {
        'tat': 0.36, 'aroma': 0.27, 'finish': 0.18,
        'body': 0.07, 'color': 0.02,
        'age': 0.05,
        'percent': 0.07  # %100 in the end.
    }
    df['custom_sim'] = (
            df['score_taste'] * weights['taste'] +
            df['score_aroma'] * weights['aroma'] +
            df['score_finish'] * weights['finish'] +
            df['score_body'] * weights['body'] +
            df['score_color'] * weights['color'] +
            df['score_age'] * weights['age'] +
            df['score_percent'] * weights['percent']
    )

    # 9) Recommendations based on Bunnahabhain reference
    if 'name' in df.columns:
        mask = df['name'].str.lower() == 'bunnahabhain'
        if mask.any():
            ref_sim = df.loc[mask, 'custom_sim'].iloc[0]
            df['sim_diff'] = (df['custom_sim'] - ref_sim).abs()
            similar5 = df[~mask].nsmallest(5, 'sim_diff')[['name','sim_diff']]
            far5   = df[~mask].nlargest(5,  'sim_diff')[['name','sim_diff']]
        else:
            similar5, far5 = pd.DataFrame(), pd.DataFrame()
    else:
        similar5, far5 = pd.DataFrame(), pd.DataFrame()

    # 10) Factor importance
    cats = ['score_taste','score_aroma','score_finish','score_body','score_color']
    corrs = df[cats+['custom_sim']].corr()['custom_sim'].abs().sort_values(ascending=False)
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(df[cats], df['custom_sim'])
    importances = pd.Series(rf.feature_importances_, index=cats).sort_values(ascending=False)

    # 11) Region × aroma profile
    region_aroma = df.groupby('region', observed=True)[aroma_feats].mean() if aroma_feats else pd.DataFrame()
    
   # 12) Average SCORE by REGION & COLOR
    avg_reg = df.groupby('region', observed=True)['score'].mean().sort_values(ascending=False) if 'region' in df.columns else pd.Series()
    avg_col = df.groupby('color', observed=True)['score'].mean().sort_values(ascending=False)  if 'color' in df.columns  else pd.Series()

    # 13) Aroma-based SCORE
    aroma_scores = []
    for feat in aroma_feats:
        means = df.groupby(feat, observed=True)['score'].mean().to_dict()
        aroma_scores.append({'aroma':feat.replace('nose_',''), 'score_if_0':means.get(0,np.nan), 'score_if_1':means.get(1,np.nan)})
    aroma_score_df = pd.DataFrame(aroma_scores)

  # 14) we can see the result.

    print("\n=== Top 5 Similar to Bunnahabhain ===")
    print(similar5.to_string(index=False))
    print("\n=== Top 5 Farthest from Bunnahabhain ===")
    print(far5.to_string(index=False))
    print("\n=== Category Correlations ===")
    print(corrs.to_frame('Correlation').to_string())
    print("\n=== RandomForest Feature Importances ===")
    print(importances.to_frame('Importance').to_string())
    print("\n=== Region × Aroma Profile ===")
    print(region_aroma.to_string())
    print("\n=== Avg SCORE by REGION ===")
    print(avg_reg.to_string())
    print("\n=== Avg SCORE by COLOR ===")
    print(avg_col.to_string())
    print("\n=== Aroma-based Avg SCORE ===")
    print(aroma_score_df.to_string(index=False))

    # 15) I generated the graphs for each of them.
    if not similar5.empty:
        similar5.set_index('name')['sim_diff'].plot(kind='bar', title='Top 5 Similar to Bunnahabhain')
        plt.tight_layout(); plt.savefig('top5_similar.png'); plt.clf()
    if not far5.empty:
        far5.set_index('name')['sim_diff'].plot(kind='bar', title='Top 5 Farthest from Bunnahabhain')
        plt.tight_layout(); plt.savefig('top5_farthest.png'); plt.clf()
    importances.plot(kind='bar', title='Feature Importances'); plt.tight_layout(); plt.savefig('feature_importances.png'); plt.clf()
    if not avg_reg.empty:
        avg_reg.plot(kind='bar', title='Average Score by Region'); plt.tight_layout(); plt.savefig('avg_score_region.png'); plt.clf()
    if not avg_col.empty:
        avg_col.plot(kind='bar', title='Average Score by Color'); plt.tight_layout(); plt.savefig('avg_score_color.png'); plt.clf()
    if not aroma_score_df.empty:
        aroma_score_df.set_index('aroma')[['score_if_0','score_if_1']].plot(kind='bar', title='Aroma-based Average Score')
        plt.tight_layout(); plt.savefig('aroma_avg_score.png'); plt.clf()

if __name__ == '__main__':
    main()
