from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["MedHouseVal"] = data.target

    for i in df.columns:
        df[i].hist(bins=30)
        plt.title(i)
        plt.show()

    df["TXPRM"] = df["MedInc"] / df["AveRooms"].replace(0, np.nan)
    df["TXLSTAT"] = df["MedInc"] / df["Population"].replace(0, np.nan)

    print(data.feature_names)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(df["TXPRM"], df["MedHouseVal"], df["AveRooms"], c=df["MedInc"], cmap="viridis")
    ax.set_xlabel('TXPRM')
    ax.set_ylabel('MedHouseVal')
    ax.set_zlabel('AveRooms')

    plt.colorbar(sc)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(df["TXLSTAT"], df["MedHouseVal"], df["AveRooms"], c=df["MedInc"], cmap="viridis")
    ax.set_xlabel('TXLSTAT')
    ax.set_ylabel('MedHouseVal')
    ax.set_zlabel('AveRooms')

    plt.colorbar(sc)
    plt.show()





if __name__ == '__main__':
    main()