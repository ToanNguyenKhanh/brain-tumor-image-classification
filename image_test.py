"""
@author: <nktoan163@gmail.com>
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from model import CNN
import argparse
import warnings
warnings.filterwarnings("ignore")

def parse_arguments():
    parser = argparse.ArgumentParser('test')
    parser.add_argument('-s', '--size', type=int, default=224, help='image size (default:224)')
    parser.add_argument('-c', '--checkpoint_path', type=str, default='trained_model/best.pt')
    parser.add_argument('-i', '--image_path', type=str,
                        default="./brain tumor.v3i.folder/test/image_dataset/normal/N_86_jpg.rf.76baad308330ec21409a2ba9271e4aea.jpg",
                        help='image path')
    args = parser.parse_args()
    return args

def test(args):
    classes = ['glioma_tumor', 'meningioma_tumor', 'normal', 'pituitary_tumor']
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    model = CNN(num_classes=len(classes)).to(device)
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        model.eval()
    else:
        print('No checkpoint')
        exit(0)
    if not args.image_path:
        print('No root path')
        exit(0)

    image = cv2.imread(args.image_path)
    image = cv2.resize(image, (args.size, args.size))
    image = np.transpose(image, (2, 0, 1))
    image = image / 255.0
    image = np.expand_dims(image, 0)
    image = torch.from_numpy(image).to(device).float()

    softmax = nn.Softmax()
    with torch.no_grad():
        prediction = model(image)
    probs = softmax(prediction)
    max_value, max_index = torch.max(probs, dim=1)
    print("This image is about {} with probability of {:.4f}".format(classes[max_index], max_value[0].item()))

    plt.figure(figsize=(10, 6))
    plt.bar(classes, probs[0].cpu().numpy(), color='skyblue')
    plt.xlabel("Class", fontsize=14)
    plt.ylabel("Probability", fontsize=14)
    plt.title(classes[max_index], fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    args = parse_arguments()
    test(args)
