import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from scipy.stats import pearsonr, spearmanr


### --- Script that calulates and plots a few statistics about volume/surface

csv_files = glob.glob(r'meta\subjects_*_filtered.csv')  # adapte le chemin si besoin

for csv_file in csv_files:
    organ=csv_file.replace("subjects_", "").replace("_filtered.csv", "")
    df=pd.read_csv(csv_file)
    

    # Filtrer les âges manquants (si besoin)
    df = df[df['age'].notnull()]
    if len(df) == 0:
        print("Aucune donnée d'âge disponible pour cet organe.")
        continue

    # Corrélation Pearson
    pearson_corr, pearson_p = pearsonr(df['age'], df['volume_cm3'])
    print(f"Corrélation Pearson (age, volume): {pearson_corr:.3f} (p={pearson_p:.2e})")

    # Corrélation Spearman (robuste aux distributions non linéaires)
    spearman_corr, spearman_p = spearmanr(df['age'], df['volume_cm3'])
    print(f"Corrélation Spearman (age, volume): {spearman_corr:.3f} (p={spearman_p:.2e})")

    # Scatterplot avec droite de régression
    plt.figure(figsize=(6,4))
    sns.regplot(data=df, x='age', y='volume_cm3', scatter_kws={'s':30}, line_kws={'color':'red'})
    plt.title(f"{organ}: Volume vs Age\nPearson r={pearson_corr:.2f}, p={pearson_p:.2e}")
    plt.xlabel("Âge")
    plt.ylabel("Volume (cm³)")
    plt.tight_layout()
    plt.show()
