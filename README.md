# IIT_simglucose

This repository provides implementations of various machine learning models to predict glucose levels using the **SimGlucose** simulator.
The models include LSTM, MLP, and RNN architectures, with a focus on parallel and tree-based approaches for training and inference.

## Features

- LSTM, MLP, and RNN models for glucose prediction.
- Custom dataset handling and training configurations.
- Tools for evaluating model performance.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/aespogom/IIT_simglucose.git
   ```
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- To train a model:
   ```bash
   python main_LSTM_parallel.py
   ```

   (Other scripts for different model architectures are available.)

## Folder Structure

- `data/` – Contains glucose data files.
- `models/` – Stores trained models.
- `results/` – Includes results from training sessions.

## Requirements

- Python 3.8+
- Required dependencies are listed in `requirements.txt`.
