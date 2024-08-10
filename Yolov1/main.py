import pandas as pd
import numpy as np
import os
from Dataset import VOCDataset

def main():
    dataset_path = '../../../DATASET/archive'
    pre_data = pd.read_csv(os.path.join(dataset_path, '8examples.csv')).values

    voc_dataset = VOCDataset(dataset_path, pre_data)
    for i in voc_dataset:
        print(i)




if __name__ == '__main__':
    main()
