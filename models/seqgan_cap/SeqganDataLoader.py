import numpy as np
import json


class FeatLoader():
    def __init__(self, img2id_json='', feat_mmp=''):
        with open(img2id_json, 'r') as fin:
            data = json.load(fin)
        self.img2id = data['img2id']
        self.imgs = data['imgs']
        self.feat_shape = tuple(data['feat_shape'])
        self.feat = np.memmap(feat_mmp, dtype='float32', mode='r', shape=self.feat_shape)

    def get_feat(self, img):
        if isinstance(img, str):
            return self.feat[self.img2id[img]]
        else:
            return self.feat[np.array([self.img2id[ii] for ii in img])]
            

    def sample(self, num):
        imgs = np.random.choice(self.imgs, num, replace=False)
        feats = self.feat[np.array([self.img2id[ii] for ii in imgs])]
        return imgs, feats


class DataLoader():
    def __init__(self, batch_size, seq_length, featloader, end_token=0):
        self.batch_size = batch_size
        self.token_stream = []
        self.seq_length = seq_length
        self.featloader = featloader
        self.end_token = end_token

    def create_batches(self, data_file):
        self.token_stream = []
        self.feat_stream = []

        with open(data_file, 'r') as raw:
            for line in raw:
                line = line.strip().split()
                img = line[0]
                parse_line = [int(x) for x in line[1:]]
                if len(parse_line) > self.seq_length:
                    self.token_stream.append(parse_line[:self.seq_length])
                    self.feat_stream.append(self.featloader.get_feat(img))
                else:
                    while len(parse_line) < self.seq_length:
                        parse_line.append(self.end_token)
                    if len(parse_line) == self.seq_length:
                        self.token_stream.append(parse_line)
                        self.feat_stream.append(self.featloader.get_feat(img))
        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.feat_stream = self.feat_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.feature_batch = np.split(np.array(self.feat_stream), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        feat = self.feature_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return feat, ret

    def reset_pointer(self):
        self.pointer = 0


class DisDataloader():
    def __init__(self, batch_size, seq_length, featloader):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])
        self.feats = np.array([])
        self.seq_length = seq_length
        self.featloader = featloader

    def load_train_data(self, positive_file, negative_file):
        # Load data
        positive_examples = []
        negative_examples = []
        imgs = []
        with open(positive_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                imgs.append(line[0])
                parse_line = [int(x) for x in line[1:]]
                if len(parse_line) == self.seq_length:
                    positive_examples.append(parse_line)
        with open(negative_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                imgs.append(line[0])
                parse_line = [int(x) for x in line[1:]]
                if len(parse_line) == self.seq_length:
                    negative_examples.append(parse_line)
        self.sentences = np.array(positive_examples + negative_examples)
        self.imgs = np.array(imgs)

        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]
        self.imgs = self.imgs[shuffle_indices]

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)
        self.imgs_batches = np.split(self.imgs, self.num_batch, 0)

        self.pointer = 0

    def next_batch(self):
        imgs = self.imgs_batches[self.pointer]
        feats = self.featloader.get_feat(imgs)
        ret = feats, self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0
