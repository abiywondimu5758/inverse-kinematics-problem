import pybullet as p
import pybullet_data
import os

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setAdditionalSearchPath(r"C:\pybullet_models\nao")  # added search path for NAO meshes
p.loadURDF("plane.urdf")
p.setGravity(0, 0, -9.81)

# Assuming your URDF and meshes are in 'C:/pybullet_models/nao/'
urdf_path = r"C:\Users\Abiy\Desktop\DS Workshop final projects\IK\ao.urdf"
robot_start_pos = [0, 0, 0.1]
robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

robot_id = p.loadURDF(urdf_path, robot_start_pos, robot_start_orientation, useFixedBase=False)

input("Press Enter to exit...")  # Keeps the window open until you press Enter
