"""
@author: <nktoan163@gmail.com>
"""
import os
import torch
import cv2
from torch.utils.data import Dataset


class braintumor_dataset(Dataset):
    def __init__(self, root='./brain tumor.v3i.folder',train=True,  transform=None):
        self.class_names = ['glioma_tumor', 'meningioma_tumor', 'normal', 'pituitary_tumor']
        if train:
            data_path = os.path.join(root, 'train')
        else:
            data_path = os.path.join(root, 'valid')

        self.images_paths = []
        self.labels = []

        for class_ in self.class_names:
            class_path = os.path.join(data_path, class_)
            for filename in os.listdir(class_path):
                image_path = os.path.join(class_path, filename)
                self.images_paths.append(image_path)
                self.labels.append(self.class_names.index(class_))

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image = cv2.imread(self.images_paths[item])
        if self.transform:
            image = self.transform(image)
        label = self.labels[item]
        return image, label

if __name__ == '__main__':
    dataset = braintumor_dataset(train=True)
    index = 123
    image, label = dataset.__getitem__(index)
    print(label)
    print(image.shape)
    # cv2.imshow("image", image)
    # cv2.waitKey()