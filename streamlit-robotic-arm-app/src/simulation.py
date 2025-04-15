def forward_kinematics(joint_angles, link_lengths):
    import numpy as np

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

def predict_joint_angles(model, scaler_X, scaler_y, input_values):
    import numpy as np

    target_input = np.array([input_values])
    target_input_scaled = scaler_X.transform(target_input)

    pred_scaled = model.predict(target_input_scaled)
    pred = scaler_y.inverse_transform(pred_scaled)[0]

    return pred[:3]  # Return only the first three joint angles

def simulate_robotic_arm(model, scaler_X, scaler_y, input_values, link_lengths):
    joint_angles = predict_joint_angles(model, scaler_X, scaler_y, input_values)
    
    num_frames = 50
    trajectory = np.linspace(np.array([0.0, 0.0, 0.0]), joint_angles, num_frames)

    positions = []
    for angles in trajectory:
        positions.append(forward_kinematics(angles, link_lengths))

    return np.array(positions)