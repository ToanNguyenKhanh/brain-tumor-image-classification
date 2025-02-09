"""
@author: <nktoan163@gmail.com>
"""
import os
import cv2
import numpy as np
import pandas as pd
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
    parser.add_argument('-r', '--root_path', type=str, default='brain tumor.v3i.folder/test/image_dataset')
    parser.add_argument('-c', '--checkpoint_path', type=str, default='trained_model/best.pt')
    parser.add_argument('-o', '--output_path', type=str, default="./Predictions_Test_Set/Prediction_image_test", help='output folder path')
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
    if not args.root_path:
        print('No root path')
        exit(0)
    data = {'Class': [], 'Predicted Class': [], 'Probability': []}

    for class_name in classes:
        class_path = os.path.join(args.root_path, class_name)
        for filename in os.listdir(class_path):
            image_path = os.path.join(class_path, filename)
            image = cv2.imread(image_path)
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
            print("This image is  {} with probability of {:.4f}".format(classes[max_index], max_value[0].item()))
            predicted_class = classes[max_index]
            probability = max_value.item()

            data['Class'].append(class_name)
            data['Predicted Class'].append(predicted_class)
            data['Probability'].append(probability)

    df = pd.DataFrame(data)
    csv_filename = os.path.join(args.output_path, 'Prediction_image_dataset.csv')
    df.to_csv(csv_filename, index=False)

    # Plot the accuracy
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10.colors
    for i, class_name in enumerate(df['Class'].unique()):
        class_df = df[df['Class'] == class_name]
        accuracy = (class_df['Class'] == class_df['Predicted Class']).mean() * 100
        plt.bar(class_name, accuracy, color=colors[i % len(colors)], label=f'{class_name} ({accuracy:.2f}%)')
        plt.text(class_name, accuracy + 1, f'{accuracy:.2f}%', ha='center', va='bottom')

    plt.title('Accuracy by Class', fontsize=18)
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    # plt.legend(fontsize=12, loc='upper right')

    plot_filename = os.path.join(args.output_path, 'accuracy_by_class.png')
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    args = parse_arguments()
    test(args)
