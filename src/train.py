import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf

def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    dense_units = params["train"]["dense_units"]
    dropout_rate = params["train"]["dropout_rate"]
    learning_rate = params["train"]["learning_rate"]
    epochs = params["train"]["epochs"]
    batch_size = params["train"]["batch_size"]

    x_train = np.load("data/processed/x_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    x_val = np.load("data/processed/x_val.npy")
    y_val = np.load("data/processed/y_val.npy")

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(dense_units, activation="relu"),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    os.makedirs("models", exist_ok=True)
    model.save("models/model.h5")

    history_df = pd.DataFrame(history.history)
    history_df.to_csv("models/history.csv", index=False)

if __name__ == "__main__":
    main()
