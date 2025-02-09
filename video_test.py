"""
@author: <nktoan163@gmail.com>
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import argparse
from model import CNN
import warnings
warnings.filterwarnings("ignore")


def parse_arguments():
    parser = argparse.ArgumentParser('test video')
    parser.add_argument('-s', '--size', type=int, default=224, help='image size (default:224)')
    parser.add_argument('-c', '--checkpoint_path', type=str, default='trained_model/best.pt', help='checkpoint path')
    parser.add_argument('-i', '--input_path', type=str, default="./brain tumor.v3i.folder/test/video_dataset/test8.mp4",
                        help='image input path')
    parser.add_argument('-o', '--output_path', type=str, default="./Predictions_Test_Set/Prediction_video_test/single_test_video_prediction.mp4",
                        help='image output path')
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
    if not args.input_path:
        print('No root path')
        exit(0)

    cap = cv2.VideoCapture(args.input_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out = cv2.VideoWriter(args.output_path, cv2.VideoWriter_fourcc(*"MJPG"), int(cap.get(cv2.CAP_PROP_FPS)),
                          (width, height))
    counter = 0
    while (cap.isOpened()):
        print(counter)
        counter += 1
        flag, frame = cap.read()
        if not flag:
            break
        image = cv2.resize(frame, (args.size, args.size))
        image = np.transpose(image, (2, 0, 1))
        image = image / 255
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(device).float()
        softmax = nn.Softmax()

        with torch.no_grad():
            prediction = model(image)
        probs = softmax(prediction)
        max_value, max_index = torch.max(probs, dim=1)  # lam theo hang
        category = classes[max_index]
        cv2.putText(frame, "Prediction:", (5, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.45, (128, 0, 128), 5,
                    cv2.LINE_AA)
        cv2.putText(frame, category, (5, height - 100 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.45, (128, 0, 128), 5,
                    cv2.LINE_AA)
        out.write(frame)
    cap.release()
    out.release()


if __name__ == '__main__':
    args = parse_arguments()
    test(args)
