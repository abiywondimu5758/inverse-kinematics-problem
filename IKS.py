import pandas as pd
import numpy as np

# === Train set ===
df_LTrain_x = pd.read_csv('arkomadataset/LeftArmDataset/LTrain_x.csv')
df_LTrain_y = pd.read_csv('arkomadataset/LeftArmDataset/LTrain_y.csv')

# === Validation set ===
df_LVal_x = pd.read_csv('arkomadataset/LeftArmDataset/LVal_x.csv')
df_LVal_y = pd.read_csv('arkomadataset/LeftArmDataset/LVal_y.csv')

# === Test set ===
df_LTest_x = pd.read_csv('arkomadataset/LeftArmDataset/LTest_x.csv')
df_LTest_y = pd.read_csv('arkomadataset/LeftArmDataset/LTest_y.csv')

# Just to see how many rows/columns you have
print("Train X shape:", df_LTrain_x.shape)
print("Train Y shape:", df_LTrain_y.shape)
print("Validation X shape:", df_LVal_x.shape)
print("Validation Y shape:", df_LVal_y.shape)
print("Test X shape:", df_LTest_x.shape)
print("Test Y shape:", df_LTest_y.shape)


X_train = df_LTrain_x.values
y_train = df_LTrain_y.values

X_val = df_LVal_x.values
y_val = df_LVal_y.values

X_test = df_LTest_x.values
y_test = df_LTest_y.values



from sklearn.preprocessing import StandardScaler

# Create scalers for inputs and outputs
scaler_X = StandardScaler()
scaler_y = StandardScaler()
print(scaler_y)
# Fit scalers only on the training data and transform training, validation, and test sets:
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

print("werwer",X_train[0], X_test_scaled[0])

y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)

print("Sample normalized input:", X_train_scaled[0])
print("Sample normalized output:", y_train_scaled[0])



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Define a simple model architecture:
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.2),  # Helps combat potential overfitting
    Dense(32, activation='relu'),
    Dense(y_train_scaled.shape[1], activation='linear')  # Linear activation for regression output
])

# Compile the model with the Mean Squared Error loss and an optimizer like Adam:
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.summary()  # View the model architecture



history = model.fit(
    X_train_scaled, y_train_scaled,
    epochs=200,           # You might adjust based on performance
    batch_size=64,
    validation_data=(X_val_scaled, y_val_scaled),
    shuffle=True
)


test_loss, test_mae = model.evaluate(X_test_scaled, y_test_scaled)
print("Test MSE:", test_loss)
print("Test MAE:", test_mae)




import matplotlib.pyplot as plt

# Plot training & validation loss values
plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('Model Mean Absolute Error')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.show()



import numpy as np

def forward_kinematics(joint_angles, link_lengths):
    """
    Computes the positions of each joint in a simple 3D robotic arm.

    Parameters:
        joint_angles: list or array of 3 angles [theta1, theta2, theta3]
        link_lengths: list or array of link lengths [l1, l2, l3]

    Returns:
        positions: A numpy array of shape (4, 3) where each row is [x, y, z]
                   starting from the base to the end-effector.
    """
    # Base position
    x0, y0, z0 = 0, 0, 0
    positions = [(x0, y0, z0)]

    # First joint: rotates in the XY plane
    theta1 = joint_angles[0]
    x1 = x0 + link_lengths[0] * np.cos(theta1)
    y1 = y0 + link_lengths[0] * np.sin(theta1)
    z1 = z0
    positions.append((x1, y1, z1))

    # Second joint: introduces vertical motion (rotation around y-axis)
    theta2 = joint_angles[1]
    x2 = x1 + link_lengths[1] * np.cos(theta1) * np.cos(theta2)
    y2 = y1 + link_lengths[1] * np.sin(theta1) * np.cos(theta2)
    z2 = z1 + link_lengths[1] * np.sin(theta2)
    positions.append((x2, y2, z2))

    # Third joint: adds another rotation (around y-axis) for demonstration
    theta3 = joint_angles[2]
    x3 = x2 + link_lengths[2] * np.cos(theta1) * np.cos(theta2 + theta3)
    y3 = y2 + link_lengths[2] * np.sin(theta1) * np.cos(theta2 + theta3)
    z3 = z2 + link_lengths[2] * np.sin(theta2 + theta3)
    positions.append((x3, y3, z3))

    return np.array(positions)



    # Generate a sequence of NN predictions from the test set (assuming at least 100 samples are present)
num_frames = 100
angles_sequence_nn = []

for i in range(num_frames):
    # Get one test sample (reshape as a single-sample batch)
    x_input = X_test_scaled[i].reshape(1, -1)
    # Predict the joint angles (scaled)
    pred_scaled = model.predict(x_input)
    # Inverse-transform to get back to the original scale
    pred = scaler_y.inverse_transform(pred_scaled)[0]
    # Use only the first three joint angles for our 3D demonstration
    joint_angles = pred[:3]
    angles_sequence_nn.append(joint_angles)

angles_sequence_nn = np.array(angles_sequence_nn)
print("NN-generated angles sequence shape:", angles_sequence_nn.shape)





import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Define the arm's link lengths (adjust these for your arm's geometry)
link_lengths = [1.0, 1.0, 0.8]

# Set up the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([-1, 3])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Robotic Arm Simulation with NN Predictions")

# Initialize line for the arm (we'll update its data in the animation)
line, = ax.plot([], [], [], 'o-', lw=4)

# Initialization function for the animation
def init():
    line.set_data([], [])
    line.set_3d_properties([])
    return (line,)

# Animation function which updates the arm position using NN predictions
def animate(i):
    joint_angles = angles_sequence_nn[i]   # Get predicted joint angles for frame i
    positions = forward_kinematics(joint_angles, link_lengths)
    xs = positions[:, 0]
    ys = positions[:, 1]
    zs = positions[:, 2]

    line.set_data(xs, ys)
    line.set_3d_properties(zs)
    return (line,)

# # Create the animation
# ani = animation.FuncAnimation(fig, animate, frames=len(angles_sequence_nn),
#                               init_func=init, blit=True, interval=50)

# plt.show()


from IPython.display import HTML
from matplotlib import rc

rc('animation', html='jshtml')  # or use 'html5' if preferred

ani = animation.FuncAnimation(fig, animate, frames=len(angles_sequence_nn),
                              init_func=init, blit=True, interval=50)

HTML(ani.to_jshtml())


# from matplotlib.animation import PillowWriter

# # Create the animation
# ani = animation.FuncAnimation(fig, animate, frames=len(angles_sequence_nn),
#                               init_func=init, blit=True, interval=50)

# # Save the animation as a GIF
# ani.save('robotic_arm_200epoch_64batchsize.gif', writer=PillowWriter(fps=20))



import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Ensure this import for 3D plotting

# --- Assume these are already defined from your previous code ---
# model, scaler_X, scaler_y, forward_kinematics, and link_lengths
# For instance, your link_lengths (for our 3-joint demo) are:
link_lengths = [1.0, 1.0, 0.8]

# Updated simulate function with animation:
def simulate():
    try:
        # Read input values from the UI (convert to float)
        x_val = float(entry_x.get())
        y_val = float(entry_y.get())
        z_val = float(entry_z.get())
        roll_val = float(entry_roll.get())
        pitch_val = float(entry_pitch.get())
        yaw_val = float(entry_yaw.get())
    except ValueError:
        print("Please enter valid numbers.")
        return

    # Create the target input vector ensuring the order matches your model's expected input
    target_input = np.array([[x_val, y_val, z_val, roll_val, pitch_val, yaw_val]])
    target_input_scaled = scaler_X.transform(target_input)

    # Predict the joint angles using your trained model
    pred_scaled = model.predict(target_input_scaled)
    pred = scaler_y.inverse_transform(pred_scaled)[0]
    # For our 3D simulation, we use only the first three joint angles
    target_joint_angles = pred[:3]
    print("Predicted joint angles:", target_joint_angles)

    # Define the initial joint angles (assumed here as [0, 0, 0]; adjust if needed)
    start_joint_angles = np.array([0.0, 0.0, 0.0])
    
    # Linearly interpolate between the start and target joint angles across many frames
    num_frames = 50
    trajectory = np.linspace(start_joint_angles, target_joint_angles, num_frames)

    # Set up the 3D plot for animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-1, 3])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Robotic Arm Animation to Target")

    # Initialize the line for the arm
    line, = ax.plot([], [], [], 'o-', lw=4)

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        return (line,)

    def animate(i):
        joint_angles = trajectory[i]
        positions = forward_kinematics(joint_angles, link_lengths)
        xs = positions[:, 0]
        ys = positions[:, 1]
        zs = positions[:, 2]
        line.set_data(xs, ys)
        line.set_3d_properties(zs)
        return (line,)
    
    ani = animation.FuncAnimation(fig, animate, frames=num_frames,
                                  init_func=init, blit=True, interval=50)
    plt.show()

# --- Build the Tkinter UI ---
root = tk.Tk()
root.title("Robotic Arm Simulation Input")

# Create labels and entry widgets for each parameter
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

# Add a button that triggers the simulation function
simulate_button = tk.Button(root, text="Simulate", command=simulate)
simulate_button.grid(row=6, column=0, columnspan=2, padx=5, pady=10)

root.mainloop()
