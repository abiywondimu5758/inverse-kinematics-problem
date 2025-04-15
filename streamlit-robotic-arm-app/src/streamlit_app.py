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
    
    # Joint 1
    theta1 = joint_angles[0]
    x1 = x0 + link_lengths[0] * np.cos(theta1)
    y1 = y0 + link_lengths[0] * np.sin(theta1)
    z1 = z0
    positions.append((x1, y1, z1))
    
    # Joint 2
    theta2 = joint_angles[1]
    x2 = x1 + link_lengths[1] * np.cos(theta1) * np.cos(theta2)
    y2 = y1 + link_lengths[1] * np.sin(theta1) * np.cos(theta2)
    z2 = z1 + link_lengths[1] * np.sin(theta2)
    positions.append((x2, y2, z2))
    
    # Joint 3
    theta3 = joint_angles[2]
    x3 = x2 + link_lengths[2] * np.cos(theta1) * np.cos(theta2 + theta3)
    y3 = y2 + link_lengths[2] * np.sin(theta1) * np.cos(theta2 + theta3)
    z3 = z2 + link_lengths[2] * np.sin(theta2 + theta3)
    positions.append((x3, y3, z3))
    
    # Joint 4
    theta4 = joint_angles[3]
    cumulative_angle_4 = theta2 + theta3 + theta4
    x4 = x3 + link_lengths[3] * np.cos(theta1) * np.cos(cumulative_angle_4)
    y4 = y3 + link_lengths[3] * np.sin(theta1) * np.cos(cumulative_angle_4)
    z4 = z3 + link_lengths[3] * np.sin(cumulative_angle_4)
    positions.append((x4, y4, z4))
    
    # Joint 5
    theta5 = joint_angles[4]
    cumulative_angle_5 = cumulative_angle_4 + theta5
    x5 = x4 + link_lengths[4] * np.cos(theta1) * np.cos(cumulative_angle_5)
    y5 = y4 + link_lengths[4] * np.sin(theta1) * np.cos(cumulative_angle_5)
    z5 = z4 + link_lengths[4] * np.sin(cumulative_angle_5)
    positions.append((x5, y5, z5))
    
    return np.array(positions)

# NEW: Differentiable forward kinematics for 5-DOF arm using TensorFlow
def forward_kinematics_tf_5(joint_angles, link_lengths):
    """
    Computes the end-effector position for a 5-DOF robotic arm.
    joint_angles: Tensor of shape [batch, 5]
    link_lengths: list of 5 link lengths
    Returns: Tensor of shape [batch, 3] representing (x, y, z)
    """
    theta1 = joint_angles[:, 0]
    theta2 = joint_angles[:, 1]
    theta3 = joint_angles[:, 2]
    theta4 = joint_angles[:, 3]
    theta5 = joint_angles[:, 4]
    
    x0 = tf.zeros_like(theta1)
    y0 = tf.zeros_like(theta1)
    z0 = tf.zeros_like(theta1)
    
    x1 = x0 + link_lengths[0] * tf.cos(theta1)
    y1 = y0 + link_lengths[0] * tf.sin(theta1)
    z1 = z0
    
    x2 = x1 + link_lengths[1] * tf.cos(theta1) * tf.cos(theta2)
    y2 = y1 + link_lengths[1] * tf.sin(theta1) * tf.cos(theta2)
    z2 = z1 + link_lengths[1] * tf.sin(theta2)
    
    x3 = x2 + link_lengths[2] * tf.cos(theta1) * tf.cos(theta2+theta3)
    y3 = y2 + link_lengths[2] * tf.sin(theta1) * tf.cos(theta2+theta3)
    z3 = z2 + link_lengths[2] * tf.sin(theta2+theta3)
    
    x4 = x3 + link_lengths[3] * tf.cos(theta1) * tf.cos(theta2+theta3+theta4)
    y4 = y3 + link_lengths[3] * tf.sin(theta1) * tf.cos(theta2+theta3+theta4)
    z4 = z3 + link_lengths[3] * tf.sin(theta2+theta3+theta4)
    
    x5 = x4 + link_lengths[4] * tf.cos(theta1) * tf.cos(theta2+theta3+theta4+theta5)
    y5 = y4 + link_lengths[4] * tf.sin(theta1) * tf.cos(theta2+theta3+theta4+theta5)
    z5 = z4 + link_lengths[4] * tf.sin(theta2+theta3+theta4+theta5)
    
    return tf.stack([x5, y5, z5], axis=1)  # shape: [batch, 3]

# NEW: Custom PINN model for 5-DOF outputs with physics loss
class PINNModel(tf.keras.Model):
    def __init__(self, link_lengths, lambda_phys=0.1, **kwargs):
        super(PINNModel, self).__init__(**kwargs)
        self.link_lengths = link_lengths  # list of 5 link lengths
        self.lambda_phys = lambda_phys    # weight for physics loss
        self.d1 = tf.keras.layers.Dense(64, activation='relu')
        self.d2 = tf.keras.layers.Dense(64, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.d3 = tf.keras.layers.Dense(32, activation='relu')
        self.out_layer = tf.keras.layers.Dense(5, activation='linear')  # output 5 joint angles
    def call(self, inputs, training=False):
        x = self.d1(inputs)
        x = self.d2(x)
        if training:
            x = self.dropout(x, training=training)
        x = self.d3(x)
        return self.out_layer(x)
    def train_step(self, data):
        x, y = data  # x: input features, y: ground truth joint angles with shape [batch, 5]
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss_data = tf.reduce_mean(tf.square(y - y_pred))
            pred_positions = forward_kinematics_tf_5(y_pred, self.link_lengths)
            target_positions = x[:, :3]  # assume x[:, :3] holds target end-effector positions
            loss_phys = tf.reduce_mean(tf.square(target_positions - pred_positions))
            loss = loss_data + self.lambda_phys * loss_phys
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss, "data_loss": loss_data, "phys_loss": loss_phys}
    def get_config(self):
        config = super(PINNModel, self).get_config()
        config.update({
            "link_lengths": self.link_lengths,
            "lambda_phys": self.lambda_phys
        })
        return config
    @classmethod
    def from_config(cls, config):
        link_lengths = config.pop("link_lengths", [1.0, 1.0, 0.8, 0.5, 0.3])
        lambda_phys = config.pop("lambda_phys", 0.1)
        return cls(link_lengths, lambda_phys, **config)

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

# --- Model Selection ---
st.sidebar.header("Model Selection")
model_type = st.sidebar.selectbox("Select Model Type", ["Neural Network", "PINN"])

# --- Build/Load Model ---
if model_type == "Neural Network":
    MODEL_PATH = "model_trained.h5"
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
        st.success("Loaded saved Neural Network model.")
    else:
        st.subheader("Neural Network Model")
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(5, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model_summary = StringIO()
        model.summary(print_fn=lambda x: model_summary.write(x + "\n"))
        st.text(model_summary.getvalue())
        st.info("Training Neural Network model... (this might take a while)")
        history = model.fit(X_train_scaled, y_train_scaled,
                            epochs=200, batch_size=64,
                            validation_data=(X_val_scaled, y_val_scaled),
                            shuffle=True, verbose=0)
        model.save(MODEL_PATH)
        st.success("Neural Network training completed and model saved.")
        test_loss, test_mae = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
        st.write("Test MSE:", test_loss)
        st.write("Test MAE:", test_mae)
else:  # PINN model selected
    PINN_MODEL_PATH = "pinn_model.keras"  # use a '.keras' extension for native Keras format
    if os.path.exists(PINN_MODEL_PATH):
        model = load_model(PINN_MODEL_PATH,
                           custom_objects={"PINNModel": PINNModel, "forward_kinematics_tf_5": forward_kinematics_tf_5},
                           safe_mode=False)
        st.success("Loaded saved PINN model.")
    else:
        st.subheader("PINN Model Training")
        # Define link lengths for 5-DOF arm
        link_lengths = [1.0, 1.0, 0.8, 0.5, 0.3]
        model = PINNModel(link_lengths=link_lengths, lambda_phys=0.1)
        model.build((None, X_train_scaled.shape[1]))  # Ensure model is built before training
        model.compile(optimizer='adam', loss=lambda y_true, y_pred: tf.constant(0.0))
        st.info("Training PINN model... (this may take a while)")
        history_pinn = model.fit(X_train_scaled, y_train_scaled,
                                 epochs=200, batch_size=64,
                                 validation_data=(X_val_scaled, y_val_scaled),
                                 verbose=0)
        model.save(PINN_MODEL_PATH)  # now saved with valid extension
        st.success("PINN training completed and model saved.")

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
else:
    st.image("training_history.png", caption="Training History (Loaded Model)")

# --- Create a sequence of NN predictions from test set ---
num_frames_test = 100
angles_sequence_nn = []
for i in range(num_frames_test):
    x_input = X_test_scaled[i].reshape(1, -1)
    pred_scaled = model.predict(x_input, verbose=0)
    pred = scaler_y.inverse_transform(pred_scaled)[0]
    joint_angles = pred[:5]  # Updated to use five joints
    angles_sequence_nn.append(joint_angles)
angles_sequence_nn = np.array(angles_sequence_nn)
st.write("NN-generated angles sequence shape:", angles_sequence_nn.shape)

# --- Animation using NN predictions ---
st.subheader("Robotic Arm Simulation with NN Predictions")
link_lengths = [1.0, 1.0, 0.8, 0.5, 0.3]  # Updated to five links

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
with st.sidebar.form(key="simulation_form"):
    st.sidebar.header("Input Parameters for Simulation")
    x_val = st.number_input("X:", value=0.0)
    y_val = st.number_input("Y:", value=0.0)
    z_val = st.number_input("Z:", value=0.0)
    roll_val = st.number_input("Roll:", value=0.0)
    pitch_val = st.number_input("Pitch:", value=0.0)
    yaw_val = st.number_input("Yaw:", value=0.0)
    submit_sim = st.form_submit_button("Simulate Robotic Arm")

if submit_sim:
    # Prepare target input and prediction
    target_input = np.array([[x_val, y_val, z_val, roll_val, pitch_val, yaw_val]])
    target_input_scaled = scaler_X.transform(target_input)
    pred_scaled = model.predict(target_input_scaled, verbose=0)
    pred = scaler_y.inverse_transform(pred_scaled)[0]
    target_joint_angles = pred[:5]  # Updated to use five joints
    st.write("Predicted Joint Angles (all five):", target_joint_angles)
    
    # Create trajectory
    start_joint_angles = np.zeros(5)  # Updated to five joints
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
        target_joint_angles = pred[:5]  # Updated to use five joints
        print("Predicted joint angles:", target_joint_angles)
        start_joint_angles = np.zeros(5)  # Updated to five joints
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