import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    test_size = params["preprocess"]["test_size"]
    seed = params["preprocess"]["seed"]

    x_train_raw = np.load("data/raw/x_train.npy")
    y_train_raw = np.load("data/raw/y_train.npy")
    x_test_raw = np.load("data/raw/x_test.npy")
    y_test = np.load("data/raw/y_test.npy")

    x_train_norm = x_train_raw.astype("float32") / 255.0
    x_test_norm = x_test_raw.astype("float32") / 255.0

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_norm,
        y_train_raw,
        test_size=test_size,
        random_state=seed
    )

    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/x_train.npy", x_train)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/x_val.npy", x_val)
    np.save("data/processed/y_val.npy", y_val)
    np.save("data/processed/x_test.npy", x_test_norm)
    np.save("data/processed/y_test.npy", y_test)

if __name__ == "__main__":
    main()
