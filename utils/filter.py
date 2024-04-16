
import os
import pandas as pd
import numpy as np

from pathlib import Path


def prepare_curves_for_gnuplot(filepath: str):
    """Prepare curves for gnuplot visualization.
    
    Parameters
    ----------
    filepath : str
        Filepath to the CSV file.
    """
    df = pd.read_csv(filepath)
    df = df.iloc[:, 1:]
    df = df.iloc[:, :-2]
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    df.to_csv(filepath.replace('.csv', '.txt'), sep=' ', index=True, header=False)


def main():
    files = os.listdir('../data/training')
    for file in files:
        if file.endswith('.csv'):
            prepare_curves_for_gnuplot(f'../data/training/{file}')
    print('Done!')


if __name__ == "__main__":
    main()
