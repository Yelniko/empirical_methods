import urllib.request
from scipy.io import arff
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def download(url):
    urllib.request.urlretrieve(url, "jm1.arff")
    data, meta = arff.loadarff("jm1.arff")
    df = pd.DataFrame(data)
    df["defects"] = df["defects"].apply(lambda x: 1 if x == b"true" else 0)
    return df

def main():
    df = download("https://raw.githubusercontent.com/ApoorvaKrisna/NASA-promise-dataset-repository/main/jm1.arff")
    corr = df.corr(numeric_only=True)

    plt.figure()
    sns.heatmap(corr)
    plt.show()

    plt.figure()
    sns.heatmap(
        corr[["defects"]].sort_values(by="defects", ascending=False),
        annot=True,
    )
    plt.show()

if __name__ == '__main__':
    main()