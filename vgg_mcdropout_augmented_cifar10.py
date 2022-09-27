import argparse
import random
from copy import deepcopy

import torch
import torch.backends
from torch import optim
from torch.hub import load_state_dict_from_url
from torch.nn import CrossEntropyLoss
from torchvision import datasets
from torchvision.models import vgg16
from torchvision.transforms import transforms
from tqdm import tqdm

from baal.active import get_heuristic, ActiveLearningDataset
from baal.active.active_loop import ActiveLearningLoop
from baal.bayesian.dropout import patch_module
from baal import ModelWrapper

import aug_lib

"""
Minimal example to use BaaL.
"""


class ExtendedActiveLearningDataset(ActiveLearningDataset):
    """A dataset that allows for active learning and working with augmentations.
    Args:
        dataset: The baseline dataset.
        labelled: An array that acts as a mask which is greater than 1 for every
            data point that is labelled, and 0 for every data point that is not
            labelled.
        make_unlabelled: The function that returns an
            unlabelled version of a datum so that it can still be used in the DataLoader.
        random_state: Set the random seed for label_randomly().
        pool_specifics: Attributes to set when creating the pool.
                                         Useful to remove data augmentation.
        last_active_steps: If specified, will iterate over the last active steps
                            instead of the full dataset. Useful when doing partial finetuning.
        augmented_map: A map that indicates which data points are augmented. 0 is for original unaugmented images. Non Zero will reference the pool index of the original image

    Notes:
        n_augmented_images_labelled: This value describes how often an augmented image was recommended by the active learning algorithm to be labelled.
                                     If you label an image with the standard label function it will also label the original label but only increase the count for unlabelled.
    """

    def __init__(
        self,
        dataset: Dataset,
        labelled: Optional[np.ndarray] = None,
        make_unlabelled: Callable = _identity,
        random_state=None,
        pool_specifics: Optional[dict] = None,
        last_active_steps: int = -1,
        augmented_map: Optional[np.ndarray] = None,
    ) -> None:
        if augmented_map is None:
            self.augmented_map = np.zeros(len(dataset), dtype=int)
        self.unaugmented_pool_size = len(dataset)
        self.n_augmented_images_labelled = 0
        self.n_unaugmented_images_labelled = 0
        super().__init__(
            dataset=dataset,
            labelled=labelled,
            make_unlabelled=make_unlabelled,
            random_state=random_state,
            pool_specifics=pool_specifics,
            last_active_steps=last_active_steps,
        )

    def can_augment(self) -> bool:
        return True

    @property
    def n_unaugmented(self):
        """The number of unaugmented data points."""
        return (~self.augmented).sum()

    @property
    def n_augmented(self):
        """The number of augmented data points."""
        return self.augmented.sum()

    @property
    def augmented(self):
        """An array that acts as a boolean mask which is True for every
        data point that is labelled, and False for every data point that is not
        labelled."""
        return self.augmented_map.astype(bool)

    def augment_n_times(self, n, augmented_dataset=None) -> None:
        """Augment the every image in the dataset n times and append those augmented images to the end of the dataset
        Currently only works if an augmented version of the dataset is already present and n=1"""
        if self.n_augmented != 0:
            raise ValueError("The dataset has already been augmented.")
        if augmented_dataset == None:
            dataset_copy = deepcopy(self._dataset)
        else:
            dataset_copy = augmented_dataset
        while n > 0:
            # print("type before"+str(type(self._dataset)))
            self.labelled_map = np.concatenate((self.labelled_map, self.labelled_map))
            self.augmented_map = np.concatenate(
                (self.augmented_map, np.arange(len(self.augmented_map)))
            )
            self._dataset = self._dataset.__add__(dataset_copy)
            # print(len(dataset_copy))
            n -= 1

    def get_augmented_ids_of_image(self, idx):
        if self.is_augmentation(idx):
            raise ValueError(
                "The idx given responds to an augmented image, please specify an id that responds to an unaugmented image!"
            )
        augmented_ids = np.where(self.augmented_map == idx)
        # print(type(augmented_ids))
        return augmented_ids

    def is_augmentation(self, idx) -> bool:
        if self.augmented_map[idx] != 0:
            return True
        else:
            return False

    def label(self, idx):
        """
        Overriding the label function of ActiveLearningDataset.
        Use this function if you want to automatically label all augmentations and the original image.
        Use label after the dataset has been augmented

        Args:
            index: one or many indices to label, relative to the pool index and not the dataset index.

        Raises:
            ValueError if the indices do not match the values or
             if no `value` is provided and `can_label` is True.
        """
        oracle_idx = self._pool_to_oracle_index(idx)
        if self.is_augmentation(oracle_idx):
            self.n_augmented_images_labelled += 1
            idx = self.augmented_map[oracle_idx]
        else:
            self.n_unaugmented_images_labelled += 1

        for id in self.get_augmented_ids_of_image(oracle_idx):
            super().label(self._oracle_to_pool_index(id))
        super().label(idx)

    def label_just_this_id(self, idx):
        """
        Can be used to label just the id provided and not the augmentations.
        It is recommended to not mix using this label function with the normal label function.
        """
        if self.is_augmentation(idx):
            self.n_augmented_images_labelled += 1
        else:
            self.n_unaugmented_images_labelled += 1
        super().label(idx)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--initial_pool", default=1000, type=int)
    parser.add_argument("--query_size", default=100, type=int)
    parser.add_argument("--lr", default=0.001)
    parser.add_argument("--heuristic", default="bald", type=str)
    parser.add_argument("--iterations", default=20, type=int)
    parser.add_argument("--shuffle_prop", default=0.05, type=float)
    parser.add_argument("--learning_epoch", default=20, type=int)
    return parser.parse_args()


def get_datasets(initial_pool):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(3 * [0.5], 3 * [0.5]),
        ]
    )
    aug_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            aug_lib.TrivialAugment(),
            transforms.ToTensor(),
            transforms.Normalize(3 * [0.5], 3 * [0.5]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
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
    active_set = ActiveLearningDataset(
        train_ds, pool_specifics={"transform": test_transform}
    )

    # We start labeling randomly.
    active_set.label_randomly(initial_pool)
    return active_set, test_set


def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    random.seed(1337)
    torch.manual_seed(1337)
    if not use_cuda:
        print("warning, the experiments would take ages to run on cpu")

    hyperparams = vars(args)

    active_set, test_set = get_datasets(hyperparams["initial_pool"])

    heuristic = get_heuristic(hyperparams["heuristic"], hyperparams["shuffle_prop"])
    criterion = CrossEntropyLoss()
    model = vgg16(pretrained=False, num_classes=10)
    weights = load_state_dict_from_url(
        "https://download.pytorch.org/models/vgg16-397923af.pth"
    )
    weights = {k: v for k, v in weights.items() if "classifier.6" not in k}
    model.load_state_dict(weights, strict=False)

    # change dropout layer to MCDropout
    model = patch_module(model)

    if use_cuda:
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=hyperparams["lr"], momentum=0.9)

    # Wraps the model into a usable API.
    model = ModelWrapper(model, criterion)

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

        val_loss = metrics["test_loss"].value
        logs = {
            "val": val_loss,
            "epoch": epoch,
            "train": metrics["train_loss"].value,
            "labeled_data": active_set.labelled,
            "Next Training set size": len(active_set),
        }
        print(logs)


if __name__ == "__main__":
    main()
