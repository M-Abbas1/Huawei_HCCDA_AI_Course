"""Load the locally stored Rock–Paper–Scissors image folders."""

from pathlib import Path

import tensorflow as tf

IMAGE_SIZE = (96, 96)
CLASS_NAMES = ("paper", "rock", "scissors")
NUM_CLASSES = len(CLASS_NAMES)
SEED = 42


def load_datasets(batch_size: int = 32, dataset_dir: str = "datasets"):
    """Load local rps/ and rps-test-set/ folders into batched datasets.

    A fixed, stratified validation subset is created from the training folder;
    the separate test folder remains reserved for final evaluation.
    """
    dataset_dir = Path(dataset_dir)
    train_dir = dataset_dir / "rps"
    test_dir = dataset_dir / "rps-test-set"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            "Expected datasets/rps/ and datasets/rps-test-set/, each containing "
            "paper/, rock/, and scissors/ class folders."
        )

    for split_dir in (train_dir, test_dir):
        missing = [name for name in CLASS_NAMES if not (split_dir / name).is_dir()]
        if missing:
            raise FileNotFoundError(f"Missing class folder(s) under {split_dir}: {missing}")

    common = dict(
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="int",
        class_names=list(CLASS_NAMES),
        color_mode="rgb",
        seed=SEED,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=0.15, subset="training", shuffle=True, **common
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=0.15, subset="validation", shuffle=True, **common
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="int",
        class_names=list(CLASS_NAMES),
        color_mode="rgb",
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)
    return train_ds, val_ds, test_ds, list(CLASS_NAMES)
