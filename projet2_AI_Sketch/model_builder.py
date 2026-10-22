"""
model_builder.py — Architecture du CNN pour la reconnaissance de croquis.
Responsable : Membre 3 (AI Engineer).

Contrat public :
    build_sketch_cnn(input_shape, num_classes, dropout_rate, learning_rate) -> tf.keras.Model
"""
import logging
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

logger = logging.getLogger(__name__)


def build_sketch_cnn(
    input_shape: tuple = (28, 28, 1),
    num_classes: int = 8,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """
    Construit et compile le CNN de classification de croquis.

    Architecture (conforme au contrat) :
        Bloc 1 : Conv2D(32) -> BN -> Conv2D(32) -> MaxPool -> Dropout(0.25)
        Bloc 2 : Conv2D(64) -> BN -> Conv2D(64) -> MaxPool -> Dropout(0.25)
        Tête   : Flatten -> Dense(256) -> BN -> Dropout(dropout_rate) -> Dense(num_classes, softmax)

    Args:
        input_shape: forme d'entrée (H, W, C). Doit être (28, 28, 1) pour Quick Draw.
        num_classes: nombre de catégories à reconnaître.
        dropout_rate: taux de dropout sur la couche dense finale.
        learning_rate: learning rate de l'optimiseur Adam.

    Returns:
        Modèle Keras compilé (Adam + categorical_crossentropy + accuracy).
    """
    model = models.Sequential(name="sketch_cnn", layers=[
        # --- Bloc convolutif 1 ---
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      input_shape=input_shape, name="conv1_1"),
        layers.BatchNormalization(name="bn1_1"),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1_2"),
        layers.MaxPooling2D((2, 2), name="pool1"),
        layers.Dropout(0.25, name="drop1"),

        # --- Bloc convolutif 2 ---
        layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2_1"),
        layers.BatchNormalization(name="bn2_1"),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2_2"),
        layers.MaxPooling2D((2, 2), name="pool2"),
        layers.Dropout(0.25, name="drop2"),

        # --- Tête de classification ---
        layers.Flatten(name="flatten"),
        layers.Dense(256, activation="relu", name="fc1"),
        layers.BatchNormalization(name="bn_fc1"),
        layers.Dropout(dropout_rate, name="drop_fc"),
        layers.Dense(num_classes, activation="softmax", name="output"),
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --- Test rapide en exécution directe ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    m = build_sketch_cnn()
    m.summary()
    params = m.count_params()
    print(f"\nNombre total de paramètres : {params:,}")
    assert 100_000 < params < 2_000_000, "Architecture hors plage attendue (~500k)"
    print("✅ model_builder.py OK")