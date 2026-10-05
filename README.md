# Fashion MNIST Classification Pipeline

An end-to-end Machine Learning pipeline for Fashion-MNIST classification using TensorFlow, DVC, and Git.

## Project Structure

- `src/prepare.py`: Downloads Fashion-MNIST dataset and stores raw arrays.
- `src/preprocess.py`: Normalizes pixel values and splits training/validation sets.
- `src/train.py`: Trains an Artificial Neural Network using configured hyperparameters.
- `src/evaluate.py`: Evaluates the model on the test set and exports metrics and confusion matrix.
- `params.yaml`: Centralized configuration for hyperparameters.
- `dvc.yaml`: DVC pipeline stage definitions.

## Execution

Execute the full pipeline:
```bash
python src/prepare.py
python src/preprocess.py
python src/train.py
python src/evaluate.py
```
