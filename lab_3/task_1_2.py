import numpy as np
from scipy import stats
import pandas as pd

def data_generate(n, m):
    data = 30 * np.random.randn(n, m) + 200
    data = np.clip(data, 50, 500)
    flat = data.flatten()
    indices = np.random.choice(flat.size, size=5, replace=False)
    flat[indices] = np.nan
    return flat.reshape(data.shape)

def task_1(data):
    print(np.nanmean(data, axis=0))
    print(np.nanmedian(data, axis=0))

    print(stats.mode(data, axis=0, nan_policy='omit').mode)
    print(stats.gmean(data, axis=0, nan_policy='omit'))

    print("Dispersion: ", np.nanvar(data, axis=0))
    print(np.nanargmax(np.nanvar(data, axis=0)))

def task_2(data):
    df = pd.DataFrame(data)
    print(df[((df[1] > 100) & (df[2] < 250))])


def main():
    data = data_generate(7, 5)
    task_1(data)
    task_2(data)

if __name__ == "__main__":
    main()