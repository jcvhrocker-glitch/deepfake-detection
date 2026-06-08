# Deepfake Detection using EfficientNet-B0

This project implements a Deepfake Detection system capable of distinguishing between real and AI-generated facial images using transfer learning with EfficientNet-B0.

## Project Workflow

Before running the Streamlit application, the notebooks must be executed in order to generate the required artifacts:

1. Download and preprocess the dataset.
2. Generate train and test splits.
3. Train the model.
4. Save the trained model in the `models/` directory.
5. Run the Streamlit application.

## Project Structure

```text
deepfake-detection/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── train.csv
│   └── test.csv
│
├── models/
│   └── best_model.keras
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_evaluation.ipynb
│
├── reports/
│   └── figures/
│
├── src/
│
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/your-username/deepfake-detection.git

cd deepfake-detection

pip install -r requirements.txt
```

## Required Preprocessing

The Streamlit application depends on files generated during notebook execution.

Run the notebooks first to create:

- Processed images
- Train/Test datasets
- Evaluation reports
- Trained model (`models/best_model.keras`)

After completing the notebooks, verify that the following files exist:

```text
models/best_model.keras
data/train.csv
data/test.csv
```

## Run the Application

```bash
streamlit run app/app.py
```

## Results

- Accuracy: 97.96%
- AUC: 0.994
- Precision: 96.09%
- Recall: 100%

## Technologies

- Python
- TensorFlow / Keras
- EfficientNet-B0
- Scikit-Learn
- Streamlit
- Computer Vision
