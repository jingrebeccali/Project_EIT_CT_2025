import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

### --- Script that calulates and plots a few statistics about volume/surface



csv_files=glob.glob(r'meta\subjects_*_filtered.csv')  # adapte le chemin si besoin

for csv_file in csv_files:
    organ=csv_file.replace("subjects_", "").replace("_filtered.csv", "")
    df=pd.read_csv(csv_file)
   
    #some statistics
    for col in ['volume_cm3', 'surface_cm2']:
        print(f"\n--{col}--")
        print(f"Moyenne : {df[col].mean():.2f}")
        print(f"Médiane : {df[col].median():.2f}")
        print(f"Ecart-type : {df[col].std():.2f}")
        print(f"Q1 (25%) : {df[col].quantile(0.25):.2f}")
        print(f"Q3 (75%) : {df[col].quantile(0.75):.2f}")

    # volume histo
    plt.figure(figsize=(7,4))
    sns.histplot(df['volume_cm3'],bins=20,kde=True,color="dodgerblue")
    plt.title(f"Histogramme du volume (cm³) -{organ}")
    plt.xlabel("Volume (cm³)")
    plt.ylabel("N sujets")
    plt.tight_layout()
    plt.show()

    # surface surface
    plt.figure(figsize=(7,4))
    sns.histplot(df['surface_cm2'],bins=20,kde=True,color="orangered")
    plt.title(f"Histogramme de la surface (cm²)-{organ}")
    plt.xlabel("Surface (cm²)")
    plt.ylabel("N sujets")
    plt.tight_layout()
    plt.show()

    #  boxplots 
    fig,axs=plt.subplots(1, 2, figsize=(10, 4))
    sns.boxplot(y=df['volume_cm3'],ax=axs[0],color="skyblue")
    axs[0].set_title("Boxplot Volume (cm³)")
    sns.boxplot(y=df['surface_cm2'],ax=axs[1], color="salmon")
    axs[1].set_title("Boxplot Surface (cm²)")
    plt.suptitle(f"Boxplots-{organ}")
    plt.tight_layout()
    plt.show()
