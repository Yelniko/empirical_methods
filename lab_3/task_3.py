import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    np.random.seed(42)

    data1 = np.random.choice(range(24), 10, replace=False)
    data2 = np.random.choice(range(24), 10, replace=False)

    s1 = pd.Series(np.random.uniform(0, 100, 10), index=data1)
    s2 = pd.Series(np.random.uniform(0, 100, 10), index=data2)

    idx1 = np.random.choice(s1.index, 2, replace=False)
    idx2 = np.random.choice(s2.index, 2, replace=False)

    s1.loc[idx1] = np.nan
    s2.loc[idx2] = np.nan

    full_index = range(24)
    s1 = s1.reindex(full_index)
    s2 = s2.reindex(full_index)

    s1 = s1.ffill()
    s2 = s2.bfill()

    fig, ax = plt.subplots()
    ax.plot(s1.index, s1.values)
    ax.plot(s2.index, s2.values)
    ax.set_xticks(range(24))
    ax.set_xlabel('Hour')
    ax.set_ylabel('CPU %')


    plt.show()


if __name__ == '__main__':
    main()