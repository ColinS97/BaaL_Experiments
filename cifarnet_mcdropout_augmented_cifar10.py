import argparse
import random
import csv
from copy import deepcopy
from time import time
import datetime

from tensorboardX import SummaryWriter
import torch
import torch.backends
from torch import optim
from torch.hub import load_state_dict_from_url
from torch.nn import CrossEntropyLoss
from torchvision import datasets
from torchvision.models import vgg16
from torchvision.transforms import transforms
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F

from baal.active import get_heuristic, ActiveLearningDataset
from baal.active.active_loop import ActiveLearningLoop
from baal.bayesian.dropout import patch_module
from baal import ModelWrapper

import aug_lib

from baal_extended.ExtendedActiveLearningDataset import ExtendedActiveLearningDataset


"""
Minimal example to use BaaL.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=50, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--initial_pool", default=1000, type=int)
    parser.add_argument("--query_size", default=50, type=int)
    parser.add_argument("--lr", default=0.001)
    parser.add_argument("--heuristic", default="bald", type=str)
    parser.add_argument("--iterations", default=20, type=int)
    parser.add_argument("--shuffle_prop", default=0.05, type=float)
    parser.add_argument("--learning_epoch", default=5, type=int)
    return parser.parse_args()


def get_datasets(initial_pool):
    transform = transforms.Compose(
        [
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(3 * [0.5], 3 * [0.5]),
        ]
    )
    aug_transform = transforms.Compose(
        [
            # transforms.Resize((224, 224)),
            aug_lib.TrivialAugment(),
            transforms.ToTensor(),
            transforms.Normalize(3 * [0.5], 3 * [0.5]),
        ]
    )
    test_transform = transforms.Compose(
        [
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(3 * [0.5], 3 * [0.5]),
        ]
    )
    # Note: We use the test set here as an example. You should make your own validation set.
    train_ds = datasets.CIFAR10(
        ".", train=True, transform=transform, target_transform=None, download=True
    )

    aug_train_ds = datasets.CIFAR10(
        ".", train=True, transform=aug_transform, target_transform=None, download=True
    )
    test_set = datasets.CIFAR10(
        ".", train=False, transform=test_transform, target_transform=None, download=True
    )
    eald_set = ExtendedActiveLearningDataset(train_ds)
    # active_set = ActiveLearningDataset(
    #    train_ds, pool_specifics={"transform": test_transform}
    # )
    eald_set.augment_n_times(2, augmented_dataset=aug_train_ds)
    # We start labeling randomly.
    eald_set.label_randomly(initial_pool)
    return eald_set, test_set


class CIFAR10Net(nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.batch1 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)

        self.pool2 = nn.MaxPool2d(2, 2)

        self.batch2 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)

        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)

        self.pool3 = nn.MaxPool2d(2, 2)

        self.batch3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 4 * 4, 128)

        self.fc2 = nn.Linear(128, 64)

        self.fc3 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        x = self.batch1(self.pool1(F.relu(self.conv2(x))))

        x = F.relu(self.conv3(x))

        x = self.batch2(self.pool2(F.relu(self.conv4(x))))

        x = F.relu(self.conv5(x))

        x = self.batch3(self.pool3(F.relu(self.conv6(x))))

        x = x.view(x.size(0), -1)

        x = self.dropout(x)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.dropout(x)

        x = self.fc3(x)
        return x


def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    random.seed(1337)
    torch.manual_seed(1337)
    if not use_cuda:
        print("warning, the experiments would take ages to run on cpu")

    now = datetime.datetime.now()
    dt_string = now.strftime("%d%m%Y%Hx%M")
    with open("metrics_cifarnet" + dt_string + ".csv", "w+", newline="") as out_file:
        csvwriter = csv.writer(out_file)
        csvwriter.writerow(
            (
                "epoch",
                "val_acc",
                "train_acc",
                "active_set.labelled",
                "active_set.n_augmented_images_labelled",
                "active_set.n_unaugmented_images_labelled",
                "len(active_set)",
            )
        )

        hyperparams = vars(args)

        active_set, test_set = get_datasets(hyperparams["initial_pool"])

        heuristic = get_heuristic(hyperparams["heuristic"], hyperparams["shuffle_prop"])
        criterion = CrossEntropyLoss()
        model = CIFAR10Net()
        # model = vgg16(pretrained=False, num_classes=10)
        # weights = load_state_dict_from_url(
        #    "https://download.pytorch.org/models/vgg16-397923af.pth"
        # )
        # weights = {k: v for k, v in weights.items() if "classifier.6" not in k}
        # model.load_state_dict(weights, strict=False)

        # change dropout layer to MCDropout
        model = patch_module(model)

        if use_cuda:
            model.cuda()
        optimizer = optim.SGD(model.parameters(), lr=hyperparams["lr"], momentum=0.9)

        # Wraps the model into a usable API.
        model = ModelWrapper(model, criterion, replicate_in_memory=False)

        logs = {}
        logs["epoch"] = 0

        # for prediction we use a smaller batchsize
        # since it is slower
        active_loop = ActiveLearningLoop(
            active_set,
            model.predict_on_dataset,
            heuristic,
            hyperparams.get("query_size", 1),
            batch_size=10,
            iterations=hyperparams["iterations"],
            use_cuda=use_cuda,
        )
        # We will reset the weights at each active learning step.
        init_weights = deepcopy(model.state_dict())

        layout = {
            "Loss/Accuracy": {
                "Loss": ["Multiline", ["loss/train", "loss/test", "loss/val"]],
                "Accuracy": [
                    "Multiline",
                    ["accuracy/train", "accuracy/test", "accuracy/val"],
                ],
            },
        }

        tensorboardwriter = SummaryWriter("tb-results" + dt_string + "/testrun")
        tensorboardwriter.add_custom_scalars(layout)

        for epoch in tqdm(range(args.epoch)):
            # Load the initial weights.
            model.load_state_dict(init_weights)
            model.train_on_dataset(
                active_set,
                optimizer,
                hyperparams["batch_size"],
                hyperparams["learning_epoch"],
                use_cuda,
            )

            # Validation!
            model.test_on_dataset(test_set, hyperparams["batch_size"], use_cuda)
            metrics = model.metrics
            should_continue = active_loop.step()
            if not should_continue:
                break

            val_acc = metrics["test_acc"].value
            logs = {
                "epoch": epoch,
                "val_acc": val_acc,
                "train_acc": metrics["train_accuracy"].value,
                "labeled_data": active_set.labelled,
                "n_augmented_images_labelled": active_set.n_augmented_images_labelled,
                "n_unaugmented_images_labelled": active_set.n_unaugmented_images_labelled,
                "Next Training set size": len(active_set),
            }
            print(logs)

            csvwriter.write(
                (
                    epoch,
                    val_acc,
                    metrics["train_accuracy"].value,
                    active_set.labelled,
                    active_set.n_augmented_images_labelled,
                    active_set.n_unaugmented_images_labelled,
                    len(active_set),
                )
            )

            tensorboardwriter.add_scalar(
                "loss/train", metrics["train_loss"].value, epoch
            )
            tensorboardwriter.add_scalar("loss/test", metrics["test_loss"].value, epoch)
            tensorboardwriter.add_scalar(
                "accuracy/train", metrics["train_accuracy"].value, epoch
            )
            tensorboardwriter.add_scalar(
                "accuracy/test", metrics["test_accuracy"].value, epoch
            )
            tensorboardwriter.add_scalar(
                "accuracy/train", metrics["validation_accuracy"].value, epoch
            )
            tensorboardwriter.add_scalar(
                "accuracy/test", metrics["validation_accuracy"].value, epoch
            )
        tensorboardwriter.close()


if __name__ == "__main__":
    main()
