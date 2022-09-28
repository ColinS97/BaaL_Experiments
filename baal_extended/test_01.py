import pytest

from ExtendedActiveLearningDataset import ExtendedActiveLearningDataset
from torchvision import datasets
from torchvision.transforms import transforms
import aug_lib


def prepare_datasets():

    aug_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            aug_lib.TrivialAugment(),
            transforms.ToTensor(),
            transforms.Normalize(3 * [0.5], 3 * [0.5]),
        ]
    )

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(3 * [0.5], 3 * [0.5]),
        ]
    )
    train_ds = datasets.CIFAR10(
        ".", train=True, transform=transform, target_transform=None, download=True
    )

    aug_train_ds = datasets.CIFAR10(
        ".", train=True, transform=aug_transform, target_transform=None, download=True
    )
    # ald_set = ActiveLearningDataset(
    #        train_ds, pool_specifics={"transform": transform}
    #    )
    eald_set = ExtendedActiveLearningDataset(train_ds)
    return eald_set, train_ds, aug_train_ds


def test_can_augment_eald():
    n_augment = 2
    n_labelled_randomly = 10
    eald_set, train_ds, aug_train_ds = prepare_datasets()
    eald_set.augment_n_times(n_augment, augmented_dataset=aug_train_ds)
    assert len(eald_set.pool) == len(train_ds) + n_augment * len(aug_train_ds)
    eald_set.label_randomly(n_labelled_randomly)
    assert len(eald_set) == n_labelled_randomly * (n_augment + 1)


if __name__ == "__main__":
    pytest.main()
