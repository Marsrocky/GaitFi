import os
import random
import numpy as np
from PIL import Image
from torch.utils import data
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn.functional as F
from tqdm import tqdm
import scipy.io as scio


def labels2cat(label_encoder, list):
    return label_encoder.transform(list)


def labels2onehot(OneHotEncoder, label_encoder, list):
    return OneHotEncoder.transform(label_encoder.transform(list).reshape(-1, 1)).toarray()


def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()


def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()


class Dataset_CRNN(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, image_path, mat_path, folders, labels, n_frames, transform=None, input_type='image'):
        "Initialization"
        self.image_path = image_path
        self.mat_path = mat_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.n_frames = n_frames
        self.input_type = input_type

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, image_path, selected_folder, use_transform):
        names = os.listdir(f'{image_path}/{selected_folder}')
        assert len(names) > 0, f'please remove the dir {image_path}/{selected_folder} where exists {len(names)} images.'

        if len(names) > self.n_frames:
            names = random.sample(names, self.n_frames)
        else:
            names += [names[-1]] * (self.n_frames - len(names))
        names = sorted(names, key=lambda info: (int(info[0:-4]), info[-4:]))

        images = []
        for name in names:
            image = Image.open(f'{image_path}/{selected_folder}/{name}')
            if use_transform is not None:
                image = use_transform(image)
            images.append(image)

        images = torch.stack(images, dim=0)
        return images

    def read_mat(self, mat_path, selected_folder):
        mat = scio.loadmat(f'{mat_path}/{selected_folder}.mat')['CSIamp']

        # normalize
        mat = (mat - 42.3199) / 4.9802

        # sampling: 2000 -> 500
        mat = mat[:, 100::3]
        mat = mat.reshape(3, 114, 500)

        # x = np.expand_dims(x, axis=0)
        mat = torch.FloatTensor(mat)
        mat = torch.tensor(mat, dtype=torch.float32)
        return mat

    def __getitem__(self, index):
        # Select sample
        folder = self.folders[index]

        # Load data
        if self.input_type == 'image':
            image = self.read_images(self.image_path, folder, self.transform)  # (input) spatial images
            mat = torch.tensor(1)
        elif self.input_type == 'mat':
            image = torch.tensor(1)
            mat = self.read_mat(self.mat_path, folder)  # (input) spatial mat
        elif self.input_type == 'both':
            image = self.read_images(self.image_path, folder, self.transform)
            mat = self.read_mat(self.mat_path, folder)

        label = torch.LongTensor([self.labels[index]])
        return image, mat, label


def train(model, device, train_loader, optimizer, metric_loss, alpha):
    model.train()

    # set model as training mode
    N_count = 0   # counting total trained sample in one epoch
    epoch_loss = 0.0
    gallery_feat, gallery_label = [], []
    for batch_idx, (image, mat, label) in enumerate(train_loader):
        # distribute data to device
        image, mat, label = image.to(device), mat.to(device), label.to(device).view(-1, )

        N_count += image.size(0)

        optimizer.zero_grad()
        hidden, output = model(image, mat)   # output has dim = (batch, number of classes)

        loss = F.cross_entropy(output, label) + metric_loss(hidden, label) * alpha
        # loss = F.cross_entropy(output, label)
        # loss = metric_loss(hidden, label)
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

    ave_loss = epoch_loss / N_count
    return ave_loss


def validation(model, device, train_loader, test_loader):
    # set model as testing mode
    model.eval()

    gallery_feat, gallery_label = [], []
    prob_feat, prob_label = [], []
    for image, mat, label in train_loader:
        # distribute data to device
        image, mat = image.to(device), mat.to(device)
        hidden, output = model(image, mat)
        gallery_feat.append(hidden)
        gallery_label.append(label)

    for image, mat, label in test_loader:
        # distribute data to device
        image, mat = image.to(device), mat.to(device)
        hidden, output = model(image, mat)
        prob_feat.append(hidden)
        prob_label.append(label)

    # return correct, total
    return gallery_feat, gallery_label, prob_feat, prob_label

def acc_calculate(gallery_feat, gallery_label, prob_feat, prob_label):
    gallery_feat = gallery_feat
    gallery_label = gallery_label.detach().cpu().numpy()
    prob_feat = prob_feat
    prob_label = prob_label.detach().cpu().numpy()
    m, n = prob_feat.shape[0], gallery_feat.shape[0]
    dist = torch.pow(prob_feat, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(gallery_feat, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, prob_feat, gallery_feat.t())
    dist = dist.cpu().detach().numpy()
    index = dist.argmin(axis=1)
    pred = np.array([gallery_label[i] for i in index])
    assert pred.shape == prob_label.shape
    total = len(pred)
    correct = np.sum((pred == prob_label).astype(np.float))
    return correct, total





