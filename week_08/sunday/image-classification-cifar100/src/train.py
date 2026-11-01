"""Train and evaluate a CNN or pretrained MobileNetV2 on CIFAR-100."""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from .data import NUM_CLASSES, load_datasets
from .model import build_cnn, build_mobilenetv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("cnn", "mobilenetv2"), default="cnn")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be a positive integer.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer.")

    tf.keras.utils.set_random_seed(42)
    train_ds, val_ds, test_ds, class_names = load_datasets(args.batch_size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "cnn":
        model = build_cnn(NUM_CLASSES)
        model_path = output_dir / "cifar100_cnn.keras"
    else:
        model = build_mobilenetv2(NUM_CLASSES)
        model_path = output_dir / "cifar100_mobilenetv2.keras"

    model.summary()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Evaluate the best validation checkpoint on the untouched test split.
    best_model = tf.keras.models.load_model(model_path)
    test_loss, test_accuracy = best_model.evaluate(test_ds, verbose=0)
    y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
    probabilities = best_model.predict(test_ds, verbose=0)
    y_pred = probabilities.argmax(axis=1)
    confusion = tf.math.confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES).numpy()

    results = {
        "model": args.model,
        "class_names": class_names,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "confusion_matrix": confusion.tolist(),
        "epochs_completed": len(history.history["loss"]),
        "best_validation_accuracy": float(max(history.history["val_accuracy"])),
    }
    (output_dir / f"{args.model}_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (output_dir / "class_names.json").write_text(
        json.dumps(class_names, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))
    print(f"Saved best model to: {model_path}")


if __name__ == "__main__":
    main()
