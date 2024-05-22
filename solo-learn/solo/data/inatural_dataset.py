import torch.utils.data as data
from PIL import Image
import os
import json
from torchvision import transforms
import random
import numpy as np


def default_loader(path):
    return Image.open(path).convert("RGB")


def load_taxonomy(ann_data, tax_levels, classes):
    # loads the taxonomy data and converts to ints
    taxonomy = {}

    if "categories" in ann_data.keys():
        num_classes = len(ann_data["categories"])
        for tt in tax_levels:
            tax_data = [aa[tt] for aa in ann_data["categories"]]
            _, tax_id = np.unique(tax_data, return_inverse=True)
            taxonomy[tt] = dict(zip(range(num_classes), list(tax_id)))
    else:
        # set up dummy data
        for tt in tax_levels:
            taxonomy[tt] = dict(zip([0], [0]))

    # create a dictionary of lists containing taxonomic labels
    classes_taxonomic = {}
    for cc in np.unique(classes):
        tax_ids = [0] * len(tax_levels)
        for ii, tt in enumerate(tax_levels):
            tax_ids[ii] = taxonomy[tt][cc]
        classes_taxonomic[cc] = tax_ids

    return taxonomy, classes_taxonomic


class INAT(data.Dataset):
    def __init__(self, root, ann_file, transform):
        # load annotations
        print("Loading annotations from: " + os.path.basename(ann_file))
        with open(ann_file) as data_file:
            ann_data = json.load(data_file)

        # set up the filenames and annotations
        self.imgs = [aa["file_name"] for aa in ann_data["images"]]
        self.ids = [aa["id"] for aa in ann_data["images"]]

        # if we dont have class labels set them to '0'
        if "annotations" in ann_data.keys():
            self.classes = [aa["category_id"] for aa in ann_data["annotations"]]
        else:
            self.classes = [0] * len(self.imgs)

        # print out some stats
        print("\t" + str(len(self.imgs)) + " images")
        print("\t" + str(len(set(self.classes))) + " classes")

        self.root = root
        self.loader = default_loader

        # augmentation params
        self.transform = transform

    def __getitem__(self, index):
        path = self.root + self.imgs[index]
        img = self.loader(path)
        species_id = self.classes[index]

        img = self.transform(img)

        return img, species_id

    def __len__(self):
        return len(self.imgs)
