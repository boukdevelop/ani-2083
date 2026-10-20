"""
train_model.py — Entraînement et export du modèle CNN.
Responsable : Membre 3 (AI Engineer).

Contrat public :
    train_and_export_model(...) -> Dict[str, Any]
"""
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")  # backend sans GUI (serveur / CI)
import matplotlib.pyplot as plt

from model_builder import build_sketch_cnn

logger = logging.getLogger(__name__)


def train_and_export_model(
    dataset_path: str,
    model_export_path: str,
    classes_export_path: str,
    categories: List[str],
    epochs: int = 20,
    batch_size: int = 128,
    input_shape: tuple = (28, 28, 1),
    learning_rate: float = 0.001,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Entraîne le CNN, sauvegarde le modèle .keras, exporte classes.json,
    et génère les courbes loss/accuracy.

    Args:
        dataset_path: chemin du .npz fourni par M2 (contient X_train, y_train, ...).
        model_export_path: où sauvegarder le .keras.
        classes_export_path: où sauvegarder la liste JSON des classes (ordre = one-hot).
        categories: liste ordonnée des noms de classes (ex. config.CATEGORIES).
        epochs: nombre max d'époques (EarlyStopping coupe avant).
        batch_size: taille de batch.
        input_shape: forme d'entrée attendue.
        learning_rate: LR initial.
        seed: graine pour la reproductibilité.

    Returns:
        dict contenant history, test_accuracy, test_loss, durée, chemins.
    """
    # ---------- 1. Reproductibilité ----------
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info(f"Seeds fixées : numpy={seed}, tensorflow={seed}")

    # ---------- 2. Chargement du dataset ----------
    logger.info(f"Chargement de {dataset_path}...")
    data = np.load(dataset_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    logger.info(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    logger.info(f"Classes : {categories}")

    # Garde-fous
    num_classes = len(categories)
    assert y_train.shape[1] == num_classes, (
        f"Incohérence : y_train a {y_train.shape[1]} colonnes, "
        f"mais {num_classes} classes attendues."
    )
    assert X_train.shape[1:] == input_shape, (
        f"Shape d'entrée incohérente : {X_train.shape[1:]} != {input_shape}"
    )

    # ---------- 3. Construction du modèle ----------
    model = build_sketch_cnn(
        input_shape=input_shape,
        num_classes=num_classes,
        learning_rate=learning_rate,
    )
    model.summary(print_fn=logger.info)

    # ---------- 4. Callbacks ----------
    Path(model_export_path).parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_export_path,
            monitor="val_loss", save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2,
            min_lr=1e-5, verbose=1,
        ),
    ]

    # ---------- 5. Entraînement ----------
    logger.info("Démarrage de l'entraînement...")
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    duree = time.time() - t0
    logger.info(f"Entraînement terminé en {duree:.1f}s")

    # ---------- 6. Évaluation sur le test ----------
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test — accuracy: {test_acc:.4f} | loss: {test_loss:.4f}")

    if test_acc < 0.90:
        logger.warning(f"⚠️ Précision {test_acc:.2%} < 90% — à investiguer.")
    else:
        logger.info(f"✅ Objectif atteint : {test_acc:.2%} > 90%")

    # ---------- 7. classes.json ----------
    Path(classes_export_path).parent.mkdir(parents=True, exist_ok=True)
    with open(classes_export_path, "w", encoding="utf-8") as f:
        json.dump(categories, f, indent=2, ensure_ascii=False)
    logger.info(f"Classes exportées : {classes_export_path}")

    # ---------- 8. Courbes d'apprentissage ----------
    _plot_learning_curves(history, model_export_path)

    return {
        "history": history.history,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "duree_secondes": round(duree, 2),
        "model_path": model_export_path,
        "classes_path": classes_export_path,
        "n_classes": num_classes,
    }


def _plot_learning_curves(history, model_path: str) -> None:
    """Génère et sauvegarde les courbes Loss/Accuracy train vs val."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plot_path = str(model_path).replace(".keras", "_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Courbes sauvegardées : {plot_path}")


# --- Test rapide en exécution directe ---
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    from config import (
        DATASET_FILE_PATH, MODEL_FILE_PATH, CLASSES_FILE_PATH, CATEGORIES,
    )

    rapport = train_and_export_model(
        dataset_path=str(DATASET_FILE_PATH),
        model_export_path=str(MODEL_FILE_PATH),
        classes_export_path=str(CLASSES_FILE_PATH),
        categories=CATEGORIES,
    )
    print("\n=== RAPPORT ===")
    for k, v in rapport.items():
        if k != "history":
            print(f"{k}: {v}")
