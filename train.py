"""
@author: <nktoan163@gmail.com>
"""
import os
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
from dataset import braintumor_dataset
from model import CNN
from tqdm.autonotebook import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Resize, Compose
import torchvision.models as models
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from sklearn.metrics import accuracy_score, confusion_matrix

import warnings

warnings.filterwarnings("ignore")


def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="viridis")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Brain tumor')
    parser.add_argument('-r', '--root', type=str, default='./brain tumor.v3i.folder')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='epochs')
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-3, help='learning')
    parser.add_argument('-s', '--image_size', type=int, default=224, help='image size')
    parser.add_argument('-c', '--checkpoint_dir', type=str, default=None, help='checkpoint directory')
    parser.add_argument('-te', '--tensorboard_dir', type=str, default='tensorboard_dir', help='tensorboard directory')
    parser.add_argument('-t', '--trained_dir', type=str, default='trained_model', help='trained directory')
    args = parser.parse_args()
    return args

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformers = Compose([
        ToTensor(),
        Resize((args.image_size, args.image_size))
    ])

    # Define train, val dataset
    train_dataset = braintumor_dataset(root='./brain tumor.v3i.folder', train=True, transform=transformers)
    val_dataset = braintumor_dataset(root='./brain tumor.v3i.folder', train=False, transform=transformers)

    # Dataloader
    train_params = {
        'batch_size': args.batch_size,
        'num_workers': 6,
        'shuffle': True,
        'drop_last': True
    }

    val_params = {
        'batch_size': args.batch_size,
        'num_workers': 6,
        'shuffle': False,
        'drop_last': False
    }

    train_dataloader = DataLoader(train_dataset, **train_params)
    val_dataloader = DataLoader(val_dataset, **val_params)

    # Model
    model = CNN(num_classes=len(train_dataset.class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

    # Load save model if existed
    if args.checkpoint_dir and os.path.isfile(args.checkpoint_dir):
        checkpoint = torch.load(args.checkpoint_dir)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint['best_accuracy']
    else:
        start_epoch = 0
        best_accuracy = 0

    # Tensorboard
    if os.path.isdir(args.tensorboard_dir):
        shutil.rmtree(args.tensorboard_dir)
    os.mkdir(args.tensorboard_dir)

    # train_model checkpoint
    if not os.path.isdir(args.trained_dir):
        os.mkdir(args.trained_dir)

    writer = SummaryWriter(args.tensorboard_dir)

    iters = len(train_dataloader)
    for epoch in range(start_epoch, args.epochs):
        # Train
        model.train()
        train_loss = []
        progress_bar = tqdm(train_dataloader, colour='green')

        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            # Forward
            predictions = model(images)
            loss_train = criterion(predictions, labels)

            # Backward
            loss_train.backward()
            optimizer.step()
            loss_train_value = loss_train.item()
            progress_bar.set_description('Epoch: {}/{}. Loss: {:4f}'.format(epoch + 1, args.epochs, loss_train_value))
            train_loss.append(loss_train_value)
            writer.add_scalar("Train/Loss", np.mean(train_loss), epoch * iters + iter)

        # Validate
        model.eval()
        loss_valid = []
        all_predictions = []
        all_ground_truths = []
        with torch.no_grad():
            for iter, (images, labels) in enumerate(val_dataloader):
                images = images.to(device)
                labels = labels.to(device)
                predictions = model(images)

                # Forward
                _, max_idx = torch.max(predictions, 1)
                loss_v = criterion(predictions, labels)
                loss_valid.append(loss_v.item())
                all_ground_truths.extend(labels.tolist())
                all_predictions.extend(max_idx.tolist())

        writer.add_scalar("Val/Loss", np.mean(loss_valid), epoch)
        accuracy = accuracy_score(all_ground_truths, all_predictions)
        writer.add_scalar("Val/Accuracy", accuracy, epoch)
        conf_matrix = confusion_matrix(all_ground_truths, all_predictions)
        plot_confusion_matrix(writer, conf_matrix, [i for i in train_dataset.class_names], epoch)

        # Dictionary checkpoint
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_accuracy": best_accuracy,
            "batch_size": args.batch_size
        }

        # Save model
        torch.save(checkpoint, os.path.join(args.trained_dir, 'last.pt'))
        if accuracy > best_accuracy:
            torch.save(checkpoint, os.path.join(args.trained_dir, 'best.pt'))
            best_accuracy = accuracy
        scheduler.step()

if __name__ == '__main__':
    args = parse_arguments()
    train(args)
