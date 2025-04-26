import numpy as np
def forward_kinematics(joint_angles, link_lengths):
    x0, y0, z0 = 0.0, 0.0, 0.0
    positions = [(x0, y0, z0)]
    theta1, theta2, theta3, theta4, theta5 = joint_angles
    # link 1
    x1 = x0 + link_lengths[0] * np.cos(theta1)
    y1 = y0 + link_lengths[0] * np.sin(theta1)
    z1 = z0
    positions.append((x1, y1, z1))
    # link 2
    x2 = x1 + link_lengths[1] * np.cos(theta1) * np.cos(theta2)
    y2 = y1 + link_lengths[1] * np.sin(theta1) * np.cos(theta2)
    z2 = z1 + link_lengths[1] * np.sin(theta2)
    positions.append((x2, y2, z2))
    # link 3
    x3 = x2 + link_lengths[2] * np.cos(theta1) * np.cos(theta2 + theta3)
    y3 = y2 + link_lengths[2] * np.sin(theta1) * np.cos(theta2 + theta3)
    z3 = z2 + link_lengths[2] * np.sin(theta2 + theta3)
    positions.append((x3, y3, z3))
    # link 4
    cum4 = theta2 + theta3 + theta4
    x4 = x3 + link_lengths[3] * np.cos(theta1) * np.cos(cum4)
    y4 = y3 + link_lengths[3] * np.sin(theta1) * np.cos(cum4)
    z4 = z3 + link_lengths[3] * np.sin(cum4)
    positions.append((x4, y4, z4))
    # link 5
    cum5 = cum4 + theta5
    x5 = x4 + link_lengths[4] * np.cos(theta1) * np.cos(cum5)
    y5 = y4 + link_lengths[4] * np.sin(theta1) * np.cos(cum5)
    z5 = z4 + link_lengths[4] * np.sin(cum5)
    positions.append((x5, y5, z5))
    return np.array(positions)