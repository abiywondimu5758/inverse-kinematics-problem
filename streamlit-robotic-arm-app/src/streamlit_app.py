import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from io import StringIO
from IPython.display import HTML
from matplotlib import rc

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler

# --- Helper: forward kinematics ---
def forward_kinematics(joint_angles, link_lengths):
    x0, y0, z0 = 0, 0, 0
    positions = [(x0, y0, z0)]
    theta1 = joint_angles[0]
    x1 = x0 + link_lengths[0] * np.cos(theta1)
    y1 = y0 + link_lengths[0] * np.sin(theta1)
    z1 = z0
    positions.append((x1, y1, z1))
    theta2 = joint_angles[1]
    x2 = x1 + link_lengths[1] * np.cos(theta1) * np.cos(theta2)
    y2 = y1 + link_lengths[1] * np.sin(theta1) * np.cos(theta2)
    z2 = z1 + link_lengths[1] * np.sin(theta2)
    positions.append((x2, y2, z2))
    theta3 = joint_angles[2]
    x3 = x2 + link_lengths[2] * np.cos(theta1) * np.cos(theta2 + theta3)
    y3 = y2 + link_lengths[2] * np.sin(theta1) * np.cos(theta2 + theta3)
    z3 = z2 + link_lengths[2] * np.sin(theta2 + theta3)
    positions.append((x3, y3, z3))
    return np.array(positions)

# --- Load dataset ---
st.header("Dataset and Training")
@st.cache_data(show_spinner=True)
def load_data():
    # Update the paths as needed
    df_LTrain_x = pd.read_csv("c:/Users/Abiy/Desktop/DS Workshop final projects/IK/arkomadataset/LeftArmDataset/LTrain_x.csv")
    df_LTrain_y = pd.read_csv("c:/Users/Abiy/Desktop/DS Workshop final projects/IK/arkomadataset/LeftArmDataset/LTrain_y.csv")
    df_LVal_x = pd.read_csv("c:/Users/Abiy/Desktop/DS Workshop final projects/IK/arkomadataset/LeftArmDataset/LVal_x.csv")
    df_LVal_y = pd.read_csv("c:/Users/Abiy/Desktop/DS Workshop final projects/IK/arkomadataset/LeftArmDataset/LVal_y.csv")
    df_LTest_x = pd.read_csv("c:/Users/Abiy/Desktop/DS Workshop final projects/IK/arkomadataset/LeftArmDataset/LTest_x.csv")
    df_LTest_y = pd.read_csv("c:/Users/Abiy/Desktop/DS Workshop final projects/IK/arkomadataset/LeftArmDataset/LTest_y.csv")
    return df_LTrain_x, df_LTrain_y, df_LVal_x, df_LVal_y, df_LTest_x, df_LTest_y

df_LTrain_x, df_LTrain_y, df_LVal_x, df_LVal_y, df_LTest_x, df_LTest_y = load_data()

st.write("Train X shape:", df_LTrain_x.shape)
st.write("Train Y shape:", df_LTrain_y.shape)
st.write("Validation X shape:", df_LVal_x.shape)
st.write("Validation Y shape:", df_LVal_y.shape)
st.write("Test X shape:", df_LTest_x.shape)
st.write("Test Y shape:", df_LTest_y.shape)

X_train = df_LTrain_x.values
y_train = df_LTrain_y.values
X_val = df_LVal_x.values
y_val = df_LVal_y.values
X_test = df_LTest_x.values
y_test = df_LTest_y.values

# --- Scale Data ---
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)

st.write("Sample normalized input:", X_train_scaled[0])
st.write("Sample normalized output:", y_train_scaled[0])

# --- Build/Load Model ---
MODEL_PATH = "model_trained.h5"
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
    st.success("Loaded saved model.")
else:
    st.subheader("Neural Network Model")
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(y_train_scaled.shape[1], activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model_summary = StringIO()
    model.summary(print_fn=lambda x: model_summary.write(x + "\n"))
    st.text(model_summary.getvalue())
    st.info("Training model... (this might take a while)")
    history = model.fit(
        X_train_scaled, y_train_scaled,
        epochs=200,
        batch_size=64,
        validation_data=(X_val_scaled, y_val_scaled),
        shuffle=True,
        verbose=0
    )
    model.save(MODEL_PATH)
    st.success("Training completed and model saved.")
    # --- Evaluate Model ---
    test_loss, test_mae = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
    st.write("Test MSE:", test_loss)
    st.write("Test MAE:", test_mae)

# --- Plot Training History ---
if 'history' in locals():
    fig1, ax = plt.subplots(1, 2, figsize=(12,5))
    ax[0].plot(history.history['loss'], label='Train Loss')
    ax[0].plot(history.history['val_loss'], label='Val Loss')
    ax[0].set_title('Model Loss (MSE)')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    
    ax[1].plot(history.history['mae'], label='Train MAE')
    ax[1].plot(history.history['val_mae'], label='Val MAE')
    ax[1].set_title('Model Mean Absolute Error')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('MAE')
    ax[1].legend()
    
    # Save the plot as an image
    plt.savefig("training_history.png")
    st.pyplot(fig1)
    st.image("training_history.png", caption="Training History (Live)")
else:
    st.image("training_history.png", caption="Training History (Loaded Model)")

# --- Create a sequence of NN predictions from test set ---
num_frames_test = 100
angles_sequence_nn = []
for i in range(num_frames_test):
    x_input = X_test_scaled[i].reshape(1, -1)
    pred_scaled = model.predict(x_input, verbose=0)
    pred = scaler_y.inverse_transform(pred_scaled)[0]
    joint_angles = pred[:3]
    angles_sequence_nn.append(joint_angles)
angles_sequence_nn = np.array(angles_sequence_nn)
st.write("NN-generated angles sequence shape:", angles_sequence_nn.shape)

# --- Animation using NN predictions ---
st.subheader("Robotic Arm Simulation with NN Predictions")
link_lengths = [1.0, 1.0, 0.8]

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_xlim([-3, 3])
ax2.set_ylim([-3, 3])
ax2.set_zlim([-1, 3])
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title("Robotic Arm Simulation with NN Predictions")

line, = ax2.plot([], [], [], 'o-', lw=4)

def init():
    line.set_data([], [])
    line.set_3d_properties([])
    return (line,)

def animate(i):
    joint_angles = angles_sequence_nn[i]
    positions = forward_kinematics(joint_angles, link_lengths)
    xs = positions[:, 0]
    ys = positions[:, 1]
    zs = positions[:, 2]
    line.set_data(xs, ys)
    line.set_3d_properties(zs)
    return (line,)

ani = animation.FuncAnimation(fig2, animate, frames=len(angles_sequence_nn),
                              init_func=init, blit=True, interval=50)
rc('animation', html='jshtml')
ani_html = ani.to_jshtml()

st.components.v1.html(ani_html, height=600)

# --- User Input Simulation via Streamlit ---
st.sidebar.header("Input Parameters for Simulation")
x_val = st.sidebar.number_input("X:", value=0.0)
y_val = st.sidebar.number_input("Y:", value=0.0)
z_val = st.sidebar.number_input("Z:", value=0.0)
roll_val = st.sidebar.number_input("Roll:", value=0.0)
pitch_val = st.sidebar.number_input("Pitch:", value=0.0)
yaw_val = st.sidebar.number_input("Yaw:", value=0.0)

if st.sidebar.button("Simulate Robotic Arm"):
    # Prepare target input and prediction
    target_input = np.array([[x_val, y_val, z_val, roll_val, pitch_val, yaw_val]])
    target_input_scaled = scaler_X.transform(target_input)
    pred_scaled = model.predict(target_input_scaled, verbose=0)
    pred = scaler_y.inverse_transform(pred_scaled)[0]
    target_joint_angles = pred[:3]
    st.write("Predicted Joint Angles (first three):", target_joint_angles)
    
    # Create trajectory
    start_joint_angles = np.array([0.0, 0.0, 0.0])
    num_frames_sim = 50
    trajectory = np.linspace(start_joint_angles, target_joint_angles, num_frames_sim)
    
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.set_xlim([-3, 3])
    ax3.set_ylim([-3, 3])
    ax3.set_zlim([-1, 3])
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title("Robotic Arm Animation to Target")

    line_sim, = ax3.plot([], [], [], 'o-', lw=4)
    def init_sim():
        line_sim.set_data([], [])
        line_sim.set_3d_properties([])
        return (line_sim,)
    def animate_sim(i):
        joint_angles = trajectory[i]
        positions = forward_kinematics(joint_angles, link_lengths)
        xs = positions[:, 0]
        ys = positions[:, 1]
        zs = positions[:, 2]
        line_sim.set_data(xs, ys)
        line_sim.set_3d_properties(zs)
        return (line_sim,)
    
    ani_sim = animation.FuncAnimation(fig3, animate_sim, frames=num_frames_sim,
                                      init_func=init_sim, blit=True, interval=50)
    ani_sim_html = ani_sim.to_jshtml()
    st.components.v1.html(ani_sim_html, height=600)
    st.sidebar.text("Simulation completed successfully.")

# --- Tkinter Simulation Code (Launch if desired) ---
def simulate_tkinter():
    import tkinter as tk
    from mpl_toolkits.mplot3d import Axes3D  # Ensure 3D plotting
    
    def simulate():
        try:
            x_val = float(entry_x.get())
            y_val = float(entry_y.get())
            z_val = float(entry_z.get())
            roll_val = float(entry_roll.get())
            pitch_val = float(entry_pitch.get())
            yaw_val = float(entry_yaw.get())
        except ValueError:
            print("Please enter valid numbers.")
            return
        
        target_input = np.array([[x_val, y_val, z_val, roll_val, pitch_val, yaw_val]])
        target_input_scaled = scaler_X.transform(target_input)
        pred_scaled = model.predict(target_input_scaled, verbose=0)
        pred = scaler_y.inverse_transform(pred_scaled)[0]
        target_joint_angles = pred[:3]
        print("Predicted joint angles:", target_joint_angles)
        start_joint_angles = np.array([0.0, 0.0, 0.0])
        num_frames_local = 50
        trajectory = np.linspace(start_joint_angles, target_joint_angles, num_frames_local)
        
        fig_local = plt.figure()
        ax_local = fig_local.add_subplot(111, projection='3d')
        ax_local.set_xlim([-3, 3])
        ax_local.set_ylim([-3, 3])
        ax_local.set_zlim([-1, 3])
        ax_local.set_xlabel('X')
        ax_local.set_ylabel('Y')
        ax_local.set_zlabel('Z')
        ax_local.set_title("Robotic Arm Animation to Target (Tkinter)")
        line_local, = ax_local.plot([], [], [], 'o-', lw=4)

        def init_local():
            line_local.set_data([], [])
            line_local.set_3d_properties([])
            return (line_local,)

        def animate_local(i):
            joint_angles = trajectory[i]
            positions = forward_kinematics(joint_angles, link_lengths)
            xs = positions[:, 0]
            ys = positions[:, 1]
            zs = positions[:, 2]
            line_local.set_data(xs, ys)
            line_local.set_3d_properties(zs)
            return (line_local,)

        ani_local = animation.FuncAnimation(fig_local, animate_local, frames=num_frames_local,
                                            init_func=init_local, blit=True, interval=50)
        plt.show()
    
    root = tk.Tk()
    root.title("Robotic Arm Simulation Input (Tkinter)")
    
    tk.Label(root, text="X:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
    entry_x = tk.Entry(root)
    entry_x.grid(row=0, column=1, padx=5, pady=5)
    
    tk.Label(root, text="Y:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
    entry_y = tk.Entry(root)
    entry_y.grid(row=1, column=1, padx=5, pady=5)
    
    tk.Label(root, text="Z:").grid(row=2, column=0, padx=5, pady=5, sticky='e')
    entry_z = tk.Entry(root)
    entry_z.grid(row=2, column=1, padx=5, pady=5)
    
    tk.Label(root, text="Roll:").grid(row=3, column=0, padx=5, pady=5, sticky='e')
    entry_roll = tk.Entry(root)
    entry_roll.grid(row=3, column=1, padx=5, pady=5)
    
    tk.Label(root, text="Pitch:").grid(row=4, column=0, padx=5, pady=5, sticky='e')
    entry_pitch = tk.Entry(root)
    entry_pitch.grid(row=4, column=1, padx=5, pady=5)
    
    tk.Label(root, text="Yaw:").grid(row=5, column=0, padx=5, pady=5, sticky='e')
    entry_yaw = tk.Entry(root)
    entry_yaw.grid(row=5, column=1, padx=5, pady=5)
    
    simulate_button = tk.Button(root, text="Simulate", command=simulate)
    simulate_button.grid(row=6, column=0, columnspan=2, padx=5, pady=10)
    
    root.mainloop()

if st.sidebar.button("Launch Tkinter Simulation"):
    st.sidebar.info("Launching Tkinter simulation in a new window.")
    simulate_tkinter()