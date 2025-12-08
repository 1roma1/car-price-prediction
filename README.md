# Car price prediction
This repository hosts a machine learning project designed to predict the sale price of cars. The project includes a complete machine learning pipeline, covering data ingestion, data processing, model training, evaluation, experiment tracking with MLflow and data versioning with DVC.

## Getting Started
Follow these steps to set up the project and run the pipelines.

### Prerequisites
You need Python 3.13 and uv packet manager installed.

### Installation
1. Clone the repository
```
git clone https://github.com/1roma1/car-price-prediction.git
```
2. Create a virtual environment and install dependencies
```
uv sync
```
3. Set up enviromental variables:
- `MLFLOW_TRACKING_URI`
- `MLFLOW_TRACKING_USERNAME` 
- `MLFLOW_TRACKING_PASSWORD`

## Machine Learning Pipeline
### 1. Data ingestion
This step handles data retriving from database.
```
python -m src.data.ingestion --date 2025-01-01
```
### 2. Data splitting
Split the data on train and test set.
```
python -m src.data.splitting
```
### 3. Data preprocessing
Clear data, input missing values and delete anomaly data.
```
python -m src.data.preprocessing
```
### 4. Data validation
```
python -m src.data.validation
```
### 5. Training model
```
python train.py --experiment ExperimentName --train-config configs/train/model.yaml
```
### 6. Evaluating model
```
python train.py --experiment ExperimentName --train-config configs/train/model.yaml --estimator estimator_id --transformer transformer_id
```
### 