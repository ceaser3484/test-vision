from torch.utils.data import Dataset
from PIL import Image
import os


class VOCDataset(Dataset):
    def __init__(self, dataset_path, pre_data):
        super().__init__()
        self.dataset_path = dataset_path
        self.pre_data = pre_data
        # print(self.pre_data.shape)



    def __len__(self):
        return self.pre_data.shape[0]

    def __getitem__(self, item):
        img, label = self.pre_data[item, :]
        image_path = os.path.join(self.dataset_path, 'images', img)
        label_path = os.path.join(self.dataset_path, 'labels', label)
        print(label_path)

        with open(label_path, 'r') as f:
            labels = f.readlines()
            # labels = [labels.rstrip('\n') for label in labels]

        return image_path, labels
