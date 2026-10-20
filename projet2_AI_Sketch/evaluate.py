"""
evaluate.py — Métriques et visualisations d'évaluation.
Responsable : Membre 3 (AI Engineer).

Contrat public :
    evaluate_model(dataset_path, model_path, classes_path, output_dir) -> dict
"""
import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

logger = logging.getLogger(__name__)


def evaluate_model(
    dataset_path: str,
    model_path: str,
    classes_path: str,
    output_dir: str = "models/eval",
) -> dict:
    """
    Évalue le modèle sur le jeu de test et génère les artefacts d'évaluation.

    Produit :
        - <output_dir>/confusion_matrix.png
        - <output_dir>/classification_report.txt
        - retourne un dict avec accuracy et métriques par classe.
    """
    # ---------- 1. Chargement ----------
    logger.info(f"Chargement du modèle {model_path}")
    model = tf.keras.models.load_model(model_path)

    with open(classes_path, "r", encoding="utf-8") as f:
        classes = json.load(f)

    data = np.load(dataset_path)
    X_test, y_test = data["X_test"], data["y_test"]
    logger.info(f"Test set: {X_test.shape}, {len(classes)} classes")

    # ---------- 2. Prédictions ----------
    y_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # ---------- 3. Métriques globales ----------
    acc = accuracy_score(y_true, y_pred)

    # ---------- 4. Métriques par classe ----------
    labels = list(range(len(classes)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    # ---------- 5. Rapport texte ----------
    report_str = classification_report(
        y_true, y_pred, target_names=classes, digits=4, zero_division=0
    )
    logger.info("\n" + report_str)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report_path = Path(output_dir) / "classification_report.txt"
    report_path.write_text(report_str, encoding="utf-8")

    # ---------- 6. Matrice de confusion ----------
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_path = _plot_confusion_matrix(cm, classes, acc, output_dir)

    # ---------- 7. Rapport structuré ----------
    per_class = {
        cls: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, cls in enumerate(classes)
    }

    # Identifier les paires les plus confondues (hors diagonale)
    confusions = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                confusions.append({
                    "vrai": classes[i],
                    "predit": classes[j],
                    "count": int(cm[i, j]),
                })
    confusions.sort(key=lambda x: x["count"], reverse=True)

    return {
        "accuracy": float(acc),
        "per_class": per_class,
        "top_confusions": confusions[:5],
        "confusion_matrix_path": str(cm_path),
        "report_path": str(report_path),
    }


def _plot_confusion_matrix(cm, classes, acc, output_dir):
    """Trace et sauvegarde la matrice de confusion."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        cbar_kws={"label": "Nombre de prédictions"},
    )
    plt.xlabel("Classe prédite")
    plt.ylabel("Classe réelle")
    plt.title(f"Matrice de confusion — Accuracy = {acc:.2%}")
    plt.tight_layout()

    path = Path(output_dir) / "confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Matrice sauvegardée : {path}")
    return path


# --- Test rapide en exécution directe ---
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    from config import (
        DATASET_FILE_PATH, MODEL_FILE_PATH, CLASSES_FILE_PATH,
    )

    rapport = evaluate_model(
        dataset_path=str(DATASET_FILE_PATH),
        model_path=str(MODEL_FILE_PATH),
        classes_path=str(CLASSES_FILE_PATH),
    )

    print("\n=== RÉSUMÉ ÉVALUATION ===")
    print(f"Accuracy globale : {rapport['accuracy']:.2%}")
    print("\nTop confusions :")
    for c in rapport["top_confusions"]:
        print(f"  {c['vrai']} → {c['predit']} : {c['count']} fois")
