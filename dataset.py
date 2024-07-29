'''
Image Classification: 6 classes (buildings, forest, glacier, mountain, sea, street)
'''
import os
import cv2
from torch.utils.data import Dataset

class NaturalSceneDataset(Dataset):
    def __init__(self, root ,train , transform=None):
        if train:
            categories_path = root + '/train'
        else:
            categories_path = root + '/valid'

        categories = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        self.all_image_paths = []
        self.all_labels = []
        self.transform = transform

        for index, category in enumerate(categories):
            category_path = categories_path + '/' + category
            for file_names in os.listdir(category_path):
                image_path = category_path + '/' + file_names
                self.all_image_paths.append(image_path)
                self.all_labels.append(index)

    def __len__(self):
        return len(self.all_labels)
    
    def __getitem__(self, idx):
        image_path = self.all_image_paths[idx]
        image = cv2.imread(image_path) # (150, 150, 3)
        if self.transform:
            image = self.transform(image)
        label = self.all_labels[idx]
        return image, label

