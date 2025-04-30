# Robotic Arm Inverse Kinematics App

## Overview
This Streamlit app trains a neural network or PINN to predict 5‑DOF joint angles for a NAO‑style robotic arm given a desired end‑effector pose, and visualizes the results via Matplotlib or PyBullet.

## Features
- Data loading & normalization
- Neural Network & PINN model training
- Physics‑informed loss for PINN
- 3D animations (Matplotlib / PyBullet)
- User‑driven simulation via Streamlit & Tkinter

## Prerequisites
- Python 3.8+
- Git (optional)
- OS with GUI support for PyBullet

## Installation

1. Clone the repo (or download files):
   ```bash
   git clone <repo-url>
   cd IK/streamlit-robotic-arm-app
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   If you don’t have `requirements.txt`, run:
   ```bash
   pip install streamlit numpy pandas matplotlib pybullet tensorflow scikit-learn imageio
   ```

4. Verify your data paths in `streamlit_app.py` point to the correct CSV files.

## Running the App

From the project root:
```bash
streamlit run streamlit_app.py
```
- Use the sidebar to select model (Neural Network or PINN) and visualization (Matplotlib or PyBullet).
- Train or load saved models (`model_trained.h5`, `pinn_model.keras`).
- Enter target pose and simulate.

## Folder Structure

```
IK/
├─ streamlit-robotic-arm-app/
│  ├─ streamlit_app.py
│  ├─ model_trained.h5          # saved NN model
│  ├─ pinn_model.keras          # saved PINN model
│  ├─ training_history.png
│  └─ README.md
└─ arkomadataset/               # dataset CSVs
   └─ LeftArmDataset/
      ├─ LTrain_x.csv
      ├─ LTrain_y.csv
      ├─ LVal_x.csv
      ├─ LVal_y.csv
      ├─ LTest_x.csv
      └─ LTest_y.csv
```

## Usage

1. Launch the Streamlit interface.
2. Select model type and visualization mode.
3. If running for the first time, training will commence (may take minutes).
4. After training/loading, view metrics and 3D plots.
5. In the sidebar form, input a target pose and click “Simulate Robotic Arm”.
6. Choose Matplotlib or PyBullet for the final animation.

## Troubleshooting

- **PyBullet GUI won’t open**: ensure your display server is running (on WSL, use an X server).
- **Missing modules**: re-run `pip install -r requirements.txt`.
- **Long training**: adjust `epochs` or reduce data size.

---

Enjoy exploring inverse kinematics with this interactive app!
