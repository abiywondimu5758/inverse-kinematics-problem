def generate_nao_left_arm_urdf(joint_angles, link_lengths):
    urdf = """<?xml version="1.0" ?>
<robot name="nao_left_arm">
  <link name="base_link">
    <visual>
      <geometry><box size="0.1 0.1 0.1"/></geometry>
      <origin xyz="0 0 0" rpy="0 0 0"/>
    </visual>
  </link>
"""
    joint_names = [
        ("LShoulderPitch", [0,0,1], "-2.0857", "2.0857"),
        ("LShoulderRoll",  [0,1,0], "-0.3142", "1.3265"),
        ("LElbowYaw",      [0,0,1], "-2.0857", "2.0857"),
        ("LElbowRoll",     [0,1,0], "-1.5621", "0.0"),
        ("LWristYaw",      [0,0,1], "-1.8238", "1.8238"),
    ]
    for i, (name, axis, lower, upper) in enumerate(joint_names):
        length = link_lengths[i]
        parent = "base_link" if i == 0 else joint_names[i-1][0] + "_link"
        origin_z = link_lengths[i-1] if i > 0 else 0
        urdf += f"""
  <link name="{name}_link">
    <visual>
      <geometry><box size="0.05 0.05 {length}"/></geometry>
      <origin xyz="0 0 {length/2}" rpy="0 0 0"/>
    </visual>
  </link>
  <joint name="{name}_joint" type="revolute">
    <parent link="{parent}"/>
    <child link="{name}_link"/>
    <origin xyz="0 0 {origin_z}" rpy="0 0 0"/>
    <axis xyz="{axis[0]} {axis[1]} {axis[2]}"/>
    <limit lower="{lower}" upper="{upper}" effort="20" velocity="1.0"/>
  </joint>
"""
    urdf += "\n</robot>\n"
    return urdf