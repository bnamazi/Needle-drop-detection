import time
import os
import copy
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import timm
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import wandb

from dataloaders import NeedleDropImageDataset, transform_train, transform_val


parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_dir",
    type=str,
    default=".\data\images",
    help="path to training directory ",
)
parser.add_argument(
    "--test_dir", type=str, default=".\data\images", help="path to test directory"
)
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--num_epochs", type=int, default=30, help="number of epochs ")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--LR", type=float, default=0.001, help="initial learning rate")
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="./runs/CNN_LR1.pth",
    help="path to the saved model",
)
parser.add_argument(
    "--left_right",
    type=bool,
    default=True,
    help="if True a multilabel classifier is detected to idebtify the probability of grabbed by the left tool and the right tool",
)
parser.add_argument(
    "--amp", type=bool, default=True, help="use mixed-precision training"
)
parser.add_argument(
    "--early_stopping",
    type=bool,
    default=True,
    help="saves only the best model if True. saves the latest checkpoint if False",
)

args = parser.parse_args()

# with torch.profiler.profile() as profiler:
#     pass

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

NUM_CLASSES = 2
NUM_WORKERS = 8


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
    wandb.init(project="Needle_drop_detection", settings=wandb.Settings(code_dir="."))

    wandb.run.log_code(
        ".", include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb")
    )

    config = wandb.config
    config.model = "efficientnet_b3"
    config.learning_rate = args.LR
    config.batch_size = args.batch_size
    config.amp = args.amp

    train_dataset = NeedleDropImageDataset(
        data_dir=args.train_dir,
        transform=transform_train,
        train=True,
        split="train",
        left_right=args.left_right,
    )

    validation_dataset = NeedleDropImageDataset(
        data_dir=args.test_dir,
        transform=transform_val,
        train=False,
        split="val",
        left_right=args.left_right,
    )

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            prefetch_factor=1,
        ),
        "val": DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
        ),
    }

    model = timm.create_model(
        model_name="efficientnet_b3", pretrained=True, num_classes=NUM_CLASSES
    )
    # print(model)
    model = model.to(args.device)

    weights = torch.FloatTensor([0.3, 0.7]).to(device=args.device)

    if args.left_right:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=0.0
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    scaler = GradScaler()

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_metric = 0.0

    # Early stpping parameters
    num_epochs_stop = 15
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(args.num_epochs):

        # Each epoch has a training and validation phase
        for split in ["train", "val"]:
            if split == "train":
                model.train()  # Set model to training mode
                dataset_length = len(train_dataset)
            else:
                model.eval()  # Set model to evaluate mode
                dataset_length = len(validation_dataset)

            running_loss = 0.0
            y_true = []
            y_pred = []

            # Iterate over data
            with tqdm(total=dataset_length) as epoch_pbar:
                epoch_pbar.set_description(f"Epoch {epoch}/{args.num_epochs - 1}")

                for batch, (image, target, path) in enumerate(dataloaders[split]):

                    inputs = image.to(args.device)
                    labels = target.to(args.device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(split == "train"):
                        if args.amp:
                            with autocast():
                                outputs = model(inputs)

                                if args.left_right:
                                    sig = nn.Sigmoid()
                                    preds = sig(outputs)
                                    loss = criterion(outputs, labels.float())
                                else:
                                    _, preds = torch.max(outputs, -1)
                                    loss = criterion(outputs, labels.long())
                        else:
                            outputs = model(inputs)

                            if args.left_right:
                                sig = nn.Sigmoid()
                                preds = sig(outputs)
                                loss = criterion(outputs, labels.float())
                            else:
                                _, preds = torch.max(outputs, -1)
                                loss = criterion(outputs, labels.long())

                        if split == "train":
                            if args.amp:
                                scaler.scale(loss).backward()
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                loss.backward()
                                optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    labels = labels.cpu().data.numpy()
                    preds = preds.cpu().data.numpy()

                    if args.left_right:
                        labels = convert_to_grabbed_dropped(labels)
                        preds = convert_to_grabbed_dropped(preds)

                    y_true = np.concatenate((labels, y_true))
                    y_pred = np.concatenate((preds, y_pred))

                    desc = (
                        f"Epoch {epoch}/{args.num_epochs - 1} - loss {loss.item():.4f}"
                    )
                    epoch_pbar.set_description(desc)
                    epoch_pbar.update(inputs.shape[0])

                    # log train loss every 10 secs
                    if split == "train":
                        if batch % 10 == 9:
                            wandb.log({f"{split} loss": loss})

            if split == "train":
                if args.amp:
                    scale = scaler.get_scale()
                    skip_lr_sched = scale != scaler.get_scale()
                    if not skip_lr_sched:
                        scheduler.step()
                else:
                    scheduler.step()

            epoch_loss = running_loss / dataset_length
            epoch_acc = accuracy_score(y_pred=y_pred, y_true=y_true)
            epoch_f1 = f1_score(y_pred=y_pred, y_true=y_true, average="binary")

            print(
                f"{split} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}".format(
                    split, epoch_loss, epoch_acc, epoch_f1
                )
            )

            if split == "val":
                wandb.log({f"{split} f1": epoch_f1})
                wandb.log({f"{split} accuracy": epoch_acc})

            if args.early_stopping:
                # save only the best model based on f1-score on validation set
                if split == "val":
                    if epoch_f1 > best_metric:
                        best_metric = epoch_f1
                        epochs_no_improve = 0
                        best_model_wts = copy.deepcopy(model.state_dict())
                        torch.save(model.state_dict(), args.checkpoint_path)
                    else:
                        epochs_no_improve += 1

                # Early stopping
                if epochs_no_improve == num_epochs_stop:
                    print("Early stopping!")
                    early_stop = True
                    break
            else:
                # save/update the model at the end of each training epoch
                if split == "train":
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), args.checkpoint_path)
        if early_stop:
            break

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val metric (F1): {best_metric:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), args.checkpoint_path)

    # test the best or latest model and log the misclassified images and confusion matrix
    y_pred = []
    y_true = []
    paths = []
    with torch.no_grad():
        for batch, (image, target, path) in enumerate(dataloaders["val"]):
            inputs = image.to(args.device)
            labels = target.to(args.device)
            path = list(path)

            if args.amp:
                with autocast():
                    outputs = model(inputs)

                    if args.left_right:
                        sig = nn.Sigmoid()
                        preds = sig(outputs)
                    else:
                        _, preds = torch.max(outputs, -1)

            else:
                outputs = model(inputs)

                if args.left_right:
                    sig = nn.Sigmoid()
                    preds = sig(outputs)
                else:
                    _, preds = torch.max(outputs, -1)

            labels = labels.cpu().data.numpy()
            preds = preds.cpu().data.numpy()

            if args.left_right:
                labels = convert_to_grabbed_dropped(labels)
                preds = convert_to_grabbed_dropped(preds)

            y_true = np.concatenate((labels, y_true))
            y_pred = np.concatenate((preds, y_pred))
            paths = paths + path

    indices = [i for i, _ in enumerate(y_true) if y_true[i] != y_pred[i]]
    table_data = []
    for i in indices:
        table_data.append([paths[i], wandb.Image(paths[i]), y_pred[i], y_true[i]])
    columns = ["id", "image", "prediction", "truth"]
    table = wandb.Table(data=table_data, columns=columns)
    wandb.log({"Missclassified images": table})

    wandb.log(
        {
            "Confusion Matrix": wandb.plot.confusion_matrix(
                y_true=y_true, preds=y_pred, class_names=["Grabbed", "Dropped"]
            ),
        }
    )


if __name__ == "__main__":
    run()
