import torch
from dataloaders import transform_val
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
import timm
import cv2
import imageio
import numpy as np

import os
import argparse


parser1 = argparse.ArgumentParser()
parser1.add_argument(
    "--video",
    type=str,
    default="./data/test_videos/caseid_000070_fps1.mp4",
    help="path to the test video ",
)
parser1.add_argument(
    "--checkpoint",
    type=str,
    default="./runs/CNN_LR1.pth",
    help="path to the trained checkpoint ",
)
parser1.add_argument(
    "--output",
    type=str,
    default="./outputs",
    help="path to the output ",
)
parser1.add_argument(
    "--left_right",
    type=bool,
    default=True,
    help="if True a multilabel classifier is detected to idebtify the probability of grabbed by the left tool and the right tool",
)
parser1.add_argument("--device", type=str, default="cuda")
parser1.add_argument(
    "--amp", type=bool, default=True, help="use mixed-precision training"
)

args = parser1.parse_args()


NUM_WORKERS = 1
BS = 1
NUM_CLASSES = 2


model = timm.create_model(
    model_name="efficientnet_b3", pretrained=True, num_classes=NUM_CLASSES
)
model.to(args.device)

model.load_state_dict(torch.load(args.checkpoint))
model.eval()

to_tensor = transforms.ToTensor()


def crop_image(img, threshold=20):
    """
    Crop the black margin out from an image
    """
    mask = img > threshold
    if img.ndim == 3:
        mask = mask.all(2)
    m, n = mask.shape
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()
    return img[row_start:row_end, col_start:col_end]


def convert_to_grabbed_dropped(labels: np.ndarray) -> np.ndarray:
    """
    Convert an array of multilabel (left, right) elements to an array of binary (grabbed/dropped)
    if not grabbed by left and right -> dropped = 1, otherwise dropped = 0
    for example: [(0,1), (0,0)] -> [0,1]
    apply threshold of 0.5 to the predictions
    """
    labels_ = [
        1 if round(labels[i][0]) == round(labels[i][1]) == 0 else 0
        for i in range(labels.shape[0])
    ]
    return labels_


def run():
    vid_name = args.video.split("/")[-1]
    cap = cv2.VideoCapture(args.video)
    frame_rate = round(cap.get(cv2.CAP_PROP_FPS))  # 1
    frame_height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    out = cv2.VideoWriter(
        f"{args.output}/{vid_name}",
        cv2.VideoWriter_fourcc(*"MPEG"),
        frame_rate,
        (frame_width, frame_height),
    )

    # cv2 deosn't read the frames
    success, frame = cap.read()

    reader = imageio.get_reader(args.video)
    for i, img in enumerate(reader):
        frame_tensor = to_tensor(
            cv2.resize(crop_image(img), (500, 500), interpolation=cv2.INTER_CUBIC)
        )
        frame_tensor = transform_val(frame_tensor)

        inputs = torch.unsqueeze(frame_tensor, 0).to(args.device)
        if args.amp:
            with autocast():
                outputs = model(inputs)

                if args.left_right:
                    sig = torch.nn.Sigmoid()
                    preds = sig(outputs)
                    preds = convert_to_grabbed_dropped(preds.cpu().detach().numpy())
                else:
                    _, preds = torch.max(outputs, -1)
                    preds = preds.cpu().detach().numpy()

        else:
            outputs = model(inputs)

            if args.left_right:
                sig = torch.nn.Sigmoid()
                preds = sig(outputs)
                preds = convert_to_grabbed_dropped(preds.cpu().detach().numpy())
            else:
                _, preds = torch.max(outputs, -1)
                preds = preds.cpu().detach().numpy()

        print(preds)

        if int(preds[0]) == 1:
            color = (0, 0, 255)
            text = "Dropped"
        else:
            color = (0, 255, 0)
            text = "Grabbed"

        cv2.putText(
            img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_4
        )
        out.write(img)

        # success, frame = cap.read()
        # count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
