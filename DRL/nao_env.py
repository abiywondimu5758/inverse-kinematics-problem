import gym, numpy as np, pybullet as p, pybullet_data
from gym import spaces
from fk_utils import forward_kinematics
from urdf_generator import generate_nao_left_arm_urdf

class NaoIKEnv(gym.Env):
    def __init__(self, poses, angles, link_lengths=[1,1,0.8,0.5,0.3]):
        super().__init__()
        self.poses = poses
        self.angles = angles
        self.link_lengths = link_lengths
        self.action_space = spaces.Box(-0.1,0.1,(5,),np.float32)
        obs_low  = np.concatenate([np.full(5,-np.pi), np.full(6,-np.inf)])
        obs_high = np.concatenate([np.full(5, np.pi), np.full(6, np.inf)])
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        if p.isConnected(): p.disconnect()
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        urdf_text = generate_nao_left_arm_urdf([0]*5, link_lengths)
        with open("nao_left.urdf","w") as f: f.write(urdf_text)
        self.robot = p.loadURDF("nao_left.urdf", [0,0,0.1], useFixedBase=True)

    def reset(self):
        idx = np.random.randint(len(self.poses))
        self.target = self.poses[idx]
        self.goal_angles = self.angles[idx]
        self.current = np.zeros(5, dtype=np.float32)
        for j,a in enumerate(self.current):
            p.resetJointState(self.robot, j, a)
        return np.concatenate([self.current, self.target])

    def step(self, action):
        self.current = np.clip(self.current + action, -np.pi, np.pi)
        for j,a in enumerate(self.current):
            p.resetJointState(self.robot, j, a)
        ee = forward_kinematics(self.current, self.link_lengths)[-1]
        dist = np.linalg.norm(ee - self.target[:3])
        r1 = -dist
        r2 = -np.linalg.norm(self.current - self.goal_angles)
        reward = r1 + 0.1*r2
        done = bool(dist < 0.02)
        obs = np.concatenate([self.current, self.target])
        return obs, reward, done, {}

    def render(self, mode='human'):
        pass