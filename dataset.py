import json
import numpy as np
from torchvision.datasets.vision import VisionDataset
import os
from random import randint
from PIL import Image
import torch


class EncodeCaption:
    """ Encoder for the captions """
    def __init__(self, captions, max_len=None):
        """:param captions: dictionnary with a list of 5 captions for each key
           :param max_len: length of the longest caption (computed if None)
        """

        self.captions = captions
        if max_len:
            self.max_len = max_len

        # If max_len is None, we find the longuest caption
        else:
            max_len = 0
            for key in captions.keys():
                for i in range(5):
                    max_len = max(max_len, len(captions[key][i]))
            self.max_len = max_len + 2

        print("_ " * 50)
        print("Word_map creation...")
        print("_ " * 50)

        word_freq = {}

        for key in self.captions.keys():
            # For each image, we have 5 captions
            for i in range(5):
                for elem in self.captions[key][i]:
                    try:
                        word_freq[elem] += 1
                    except:
                        word_freq[elem] = 1

        words = word_freq.keys()
        self.word_map = {k: v + 1 for v, k in enumerate(words)}
        self.word_map['<start>'] = len(self.word_map) + 1
        self.word_map['<end>'] = len(self.word_map) + 1
        self.word_map['<pad>'] = 0

        print("_ " * 50)
        print("Word_map created!")
        print("_ " * 50)

    def encode_word(self, cap):
        """Encode the captions cap
           :param cap: the caption pre-processed
           :return the encoded corresponding caption"""

        encoded_cap = np.zeros(self.max_len)
        encoded_cap[0] = self.word_map['<start>']

        for i, word in enumerate(cap):
            encoded_cap[i + 1] = self.word_map[word]

        encoded_cap[len(cap) + 1] = self.word_map['<end>']
        return encoded_cap, len(cap) + 2


class CocoDataset(VisionDataset):
    """ Dataset class that handle COCO caption dataset """
    def __init__(self, img_dir, encoded_caption_dir, data_train=True, n_samples=10000,
                 transform=None, target_transform=None, transforms=None):
        """ :param img_dir: str containing the path to the image folder
            :param encoded_caption_dir: str containing the path to the caption json file
            :param data_train: Boolean whether we want the training or validation data (2/3 1/3 split)
            :param n_samples: int if the dataset is too big, you can limit the number of samples
            :param transform: transformation to do on the images
            :param target_transform: transformation to do on the encoded captions
            """
        super(CocoDataset, self).__init__(img_dir, transforms, transform, target_transform)

        self.encoded_caption_dir = encoded_caption_dir
        self.maximum_length = n_samples
        self.img_dir = img_dir

        # Captions opening
        with open(self.encoded_caption_dir) as f:
            d = json.load(f)['annotations']

        self.captions = {}
        for dico in d:
            cap = dico['caption'].lower().split()
            if cap[-1] == '':
                cap = cap[:-1]
            if cap[-1][-1] == '.':
                cap[-1] = cap[-1][:-1]
            try:
                if len(self.captions[dico['image_id']]) < 5:
                    self.captions[dico['image_id']].append(cap)
            except:
                self.captions[dico['image_id']] = [cap]

        # Images paths
        self.img_dirs = [os.path.join(self.img_dir, i) for i in os.listdir(self.img_dir)]
        self.img_dirs = self.img_dirs[:min(self.maximum_length, len(self.img_dirs))]
        if data_train:
            self.img_dirs = self.img_dirs[:(len(self.img_dirs) * 2) // 3]
        else:
            self.img_dirs = self.img_dirs[(len(self.img_dirs) * 2) // 3:]
        # Captions encoder
        self.encoder = EncodeCaption(self.captions)

    def __getitem__(self, index):
        """:param index (int): Index
           :return tuple: Tuple (image, target). target is a list of captions for the image.
        """
        path = self.img_dirs[index]
        # Caption opening
        img_id = int(path[-16:-4])
        anns = self.captions[img_id]
        all_cap = []
        for x in anns:
            all_cap.append(self.encoder.encode_word(x)[0])
        # We pick one caption out of the 5 at random
        caption_number = randint(0, 4)
        ann = self.captions[img_id][caption_number]
        # Caption encoding
        encoded_ann, len_cap = self.encoder.encode_word(ann)

        # Image opening and transformation
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            encoded_ann = self.target_transform(encoded_ann)
            all_cap = self.target_transform(all_cap)

        return img, ann, encoded_ann, torch.LongTensor([len_cap]), all_cap

    def __len__(self):
        return len(self.img_dirs)
