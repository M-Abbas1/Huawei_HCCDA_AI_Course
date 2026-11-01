"""Load and prepare the CIFAR-100 fine-label dataset."""

from pathlib import Path

import tensorflow as tf

IMAGE_SIZE = (32, 32)
CLASS_NAMES = (
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee",
    "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup",
    "dinosaur", "dolphin", "elephant", "flatfish", "forest", "fox", "girl",
    "hamster", "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard",
    "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree",
    "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
    "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
)

NUM_CLASSES = len(CLASS_NAMES)
SEED = 42
VALIDATION_SIZE = 5_000

DATASET_DIR = Path(__file__).resolve().parents[1] / "datasets" / "CIFAR-100-dataset"


def load_datasets(batch_size: int = 32):
    """Load CIFAR-100 from ``datasets/CIFAR-100-dataset``.

    Ten percent of the local training images are used for validation. The
    provided test split remains reserved for final evaluation.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    train_dir = DATASET_DIR / "train"
    test_dir = DATASET_DIR / "test"
    
    train_images = tf.keras.utils.image_dataset_from_directory(
        str(train_dir),
        class_names=list(CLASS_NAMES),
        image_size=IMAGE_SIZE,
        batch_size=None,
        shuffle=False,
    )
    train_images = train_images.shuffle(
        50_000, seed=SEED, reshuffle_each_iteration=False
    )
    train_count = 50_000 - VALIDATION_SIZE
    train_ds = train_images.take(train_count).shuffle(
        10_000, seed=SEED, reshuffle_each_iteration=True
    ).batch(batch_size)
    val_ds = train_images.skip(train_count).batch(batch_size)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        str(test_dir),
        labels="inferred",
        label_mode="int",
        class_names=list(CLASS_NAMES),
        color_mode="rgb",
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)
    return train_ds, val_ds, test_ds, list(CLASS_NAMES)
