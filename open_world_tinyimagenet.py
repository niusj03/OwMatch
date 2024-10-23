from torch.utils.data import Dataset, DataLoader
from torchvision import models, utils, datasets, transforms
import numpy as np
import sys
import os
from PIL import Image


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None, labeled=True, labeled_num=100, 
                 labeled_ratio=0.5, rand_number=0, unlabeled_idxs=None, ibf=1):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        if labeled:
            labeled_classes = range(labeled_num)
            self.labeled_idxs, self.unlabeled_idxs = self.get_label_index(labeled_classes, labeled_ratio, ibf, num_class=200)
            self.shrink_data(self.labeled_idxs)
        else:
            if train:
                assert unlabeled_idxs is not None
                # import pdb; pdb.set_trace()
                self.shrink_data(unlabeled_idxs)
            else:  # val
                self.shrink_data(range(len(self.images)))









        
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        # load the class names
        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        # mapping to classname, not necessary
        words_file = os.path.join(self.root_dir, "words.txt")
        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

        

    # return classnames, not necessary
    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]
    
    def get_label_index(self, labeled_classes, labeled_ratio, ibf=1, num_class=200):
        import math
        labeled_idxs = []
        unlabeled_idxs = []
        labels = np.array([i[1] for i in self.images])

        labeled_num_each_class = []
        unlabeled_num_each_class = []

        for i in range(num_class):
            idx = np.where(labels == i)[0]
            np.random.shuffle(idx)
            img_max = len(idx)
            num = img_max * ((1 / ibf) ** (i / (num_class - 1.0)))
            idx = idx[:int(num)]
            lbl_num = math.ceil(labeled_ratio * len(idx))
            if i in labeled_classes:
                labeled_idxs.extend(idx[:int(lbl_num)])
                unlabeled_idxs.extend(idx[int(lbl_num):])
                labeled_num_each_class.append(len(idx[:int(lbl_num)]))
                unlabeled_num_each_class.append(len(idx[int(lbl_num):]))
            else:
                unlabeled_idxs.extend(idx)
                labeled_num_each_class.append(0)
                unlabeled_num_each_class.append(len(idx))

        return labeled_idxs, unlabeled_idxs

    def shrink_data(self, idx):
        # import pdb; pdb.set_trace()
        tgt = np.array([i[1] for i in self.images])
        data = [self.images[i] for i in idx]
        self.data = [data[i][0] for i in range(len(data))]
        self.targets = tgt[idx].tolist()
        self.len_dataset = len(self.data)
        

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.data[idx], self.targets[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt

tinyimagenet_mean, tinyimagenet_std = (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
dict_transform = {

    'tinyimagenet_test': transforms.Compose([
        transforms.Resize(64), # add for img
        transforms.CenterCrop(64), # add for img
        transforms.ToTensor(),
        transforms.Normalize((tinyimagenet_mean), (tinyimagenet_std))
    ])
}
