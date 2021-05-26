#!/usr/bin/env python3


"""
@author: xi
"""

import argparse
import os
from typing import Callable

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

import clr
import dataset
from utils import CosineWarmUpAnnealingLR


class Trainer(object):

    def __init__(self,
                 model: nn.Module,
                 emb_size: int,
                 proj_head_fn: Callable[[int, int], nn.Module],
                 proj_size,
                 loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 max_lr: float,
                 weight_decay: float,
                 num_loops: int,
                 optimizer: str,
                 device: str):
        self._model = model.to(device)
        self._proj_head = proj_head_fn(emb_size, proj_size).to(device)
        self._loss_fn = loss_fn
        self._device = device

        self._parameters = [*self._model.parameters(), *self._proj_head.parameters()]
        optimizer_class = getattr(optim, optimizer)
        self._optimizer = optimizer_class(self._parameters, lr=max_lr, weight_decay=weight_decay)
        self._scheduler = CosineWarmUpAnnealingLR(self._optimizer, num_loops)

    def predict(self, x):
        with torch.no_grad():
            x = x.to(self._device)
            h = self._model(x)
            return h.detach().cpu()

    def train(self, x1, x2):
        x1 = x1.to(self._device)
        x2 = x2.to(self._device)

        z1 = self._proj_head(self._model(x1))
        z2 = self._proj_head(self._model(x2))
        loss = self._loss_fn(z1, z2)

        loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()
        self._scheduler.step()
        return loss.detach().cpu(), self._scheduler.get_last_lr()[0]


def evaluate(trainer: Trainer,
             train_loader: DataLoader,
             test_loader: DataLoader):
    # get train embeddings
    feature_list, label_list = [], []
    loop = tqdm(train_loader, leave=False, dynamic_ncols=True, desc='Testing')
    for doc in loop:
        feature = trainer.predict(doc['feature']).numpy()
        label = doc['label'].numpy()
        feature_list.extend(feature)
        label_list.extend(label)
    train_feature = np.array(feature_list)
    train_label = np.array(label_list)

    # get test embeddings
    feature_list, label_list = [], []
    loop = tqdm(test_loader, leave=False, dynamic_ncols=True, desc='Testing')
    for doc in loop:
        feature = trainer.predict(doc['feature']).numpy()
        label = doc['label'].numpy()
        feature_list.extend(feature)
        label_list.extend(label)
    test_feature = np.array(feature_list)
    test_label = np.array(label_list)

    # normalize the features
    mean = np.mean(train_feature, 0, keepdims=True)
    sigma = np.sqrt(np.var(train_feature, 0, keepdims=True))
    train_feature = (train_feature - mean) / (sigma + 1e-10)
    test_feature = (test_feature - mean) / (sigma + 1e-10)

    # perform the classification through LR
    classifier = LogisticRegression(max_iter=10000)
    classifier.fit(train_feature, train_label)
    pred_label = classifier.predict(test_feature)
    acc = accuracy_score(test_label, pred_label)
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--max-lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.3)
    parser.add_argument('--optimizer', default='AdamW')
    parser.add_argument('--base-model', default='resnet18')
    parser.add_argument('--emb-size', type=int, default=512)
    parser.add_argument('--proj-size', type=int, default=128)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model_class = getattr(models, args.base_model)
    model = model_class(pretrained=False, num_classes=args.emb_size)
    print('Model created.')

    train_path = os.path.join(args.data_path, 'train.ds')
    test_path = os.path.join(args.data_path, 'test.ds')
    unlabeled_path = os.path.join(args.data_path, 'unlabeled.ds')
    unlabeled_loader = DataLoader(
        dataset.create_unsupervised_dataset(
            [train_path, unlabeled_path] if os.path.exists(unlabeled_path)
            else [train_path]
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=20,
        pin_memory=True,
        drop_last=True
    )
    train_loader = DataLoader(
        dataset.create_supervised_dataset([train_path]),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=5
    )
    test_loader = DataLoader(
        dataset.create_supervised_dataset([test_path]),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=5
    )
    print('Data loaded.')

    trainer = Trainer(
        model=model,
        emb_size=args.emb_size,
        proj_head_fn=clr.ProjectionHead,
        proj_size=args.proj_size,
        loss_fn=clr.nt_xent_loss,
        max_lr=args.max_lr,
        weight_decay=args.weight_decay,
        num_loops=args.num_epochs * len(unlabeled_loader),
        optimizer=args.optimizer,
        device='cuda'
    )

    loss_g = 0.0
    for epoch in range(args.num_epochs):
        # train one epoch
        model.train()
        loop = tqdm(unlabeled_loader, leave=False, dynamic_ncols=True)
        for doc in loop:
            x1, x2 = doc['feature']
            loss, lr = trainer.train(x1, x2)
            loss = float(loss.numpy())
            loss_g = 0.9 * loss_g + 0.1 * loss
            loop.set_description(f'[{epoch + 1}/{args.num_epochs}] L={loss_g:.06f} lr={lr:.01e}', False)

        # evaluate for every n epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            acc = evaluate(trainer, train_loader, test_loader)
            tqdm.write(f'[{epoch + 1}/{args.num_epochs}] L={loss_g:.06f} Acc={acc:.02%}')
        else:
            tqdm.write(f'[{epoch + 1}/{args.num_epochs}] L={loss_g:.06f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
