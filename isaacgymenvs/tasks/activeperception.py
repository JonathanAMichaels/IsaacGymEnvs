import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from .base.vec_task import VecTask

import cv2  # importing for visualization

class activeperception(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.reset_dist = self.cfg["env"]["resetDist"]

        self.max_push_effort = self.cfg["env"]["maxEffort"]
        self.max_episode_length = 500

        self.cfg["env"]["numObservations"] = 6
        self.cfg["env"]["numActions"] = 3

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)

        self.gym.viewer_camera_look_at(
            self.viewer, None, gymapi.Vec3(15.0, 15.0, 15.0), gymapi.Vec3(0, 0, 0))

    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis = self.cfg["sim"]["up_axis"]
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        #self._create_ground_plane()  # no need for a ground plane
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # define plane on which environments are initialized
        lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/rotating_rectangle.urdf"  # we can randomize this for every separate environment

        #if "asset" in self.cfg["env"]:
        #    asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
        #    asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(asset)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

        self.object_handles = []
        self.camera_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            actor_handle = self.gym.create_actor(env_ptr, asset, pose, "object", i, 1)

            props = self.gym.get_actor_dof_properties(env_ptr, actor_handle)
            props["driveMode"].fill(gymapi.DOF_MODE_VEL)  # set control mode as velocity control
            props["stiffness"].fill(0.0)
            props["damping"].fill(1.0)
            self.gym.set_actor_dof_properties(env_ptr, actor_handle, props)

            # add camera sensors to capture the scene
            camera_props = gymapi.CameraProperties()
            camera_props.width = 128
            camera_props.height = 128
            camera_props.horizontal_fov = 60
            #camera_props.enable_tensors = True  # we'll enable this when the images are passed directly to a neural network on the GPU
            camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)
            pos = np.random.rand(3)  # randomize camera position
            pos = pos / np.linalg.norm(pos) * 1.5  # scale camera position so that it lies on a sphere
            self.gym.set_camera_location(camera_handle, env_ptr, gymapi.Vec3(pos[0], pos[1], pos[2]), gymapi.Vec3(0, 0, 0))

            self.envs.append(env_ptr)
            self.object_handles.append(actor_handle)
            self.camera_handles.append(camera_handle)

    def compute_reward(self):
        # retrieve environment observations from buffer
        pole_angle = self.obs_buf[:, 0]
        pole_vel = self.obs_buf[:, 1]
        cart_vel = self.obs_buf[:, 0]
        cart_pos = self.obs_buf[:, 1]

        self.rew_buf[:], self.reset_buf[:] = compute_cartpole_reward(
            pole_angle, pole_vel, cart_vel, cart_pos,
            self.reset_dist, self.reset_buf, self.progress_buf, self.max_episode_length
        )

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)

        self.obs_buf[env_ids, 0] = self.dof_pos[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 1] = self.dof_vel[env_ids, 0].squeeze()

        self.gym.render_all_camera_sensors(self.sim)
        for i in env_ids:
            # Use camera tensor when you want to keep the image on the GPU to pass it to a model
            #camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_COLOR)
            camera_image = self.gym.get_camera_image(self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_COLOR)
            camera_image = camera_image.reshape((camera_image.shape[0], camera_image.shape[0], 4))
            camera_image = camera_image[:, :, :3]
            camera_image = cv2.cvtColor(camera_image, cv2.COLOR_RGB2BGR)
            cv2.imshow('Camera Observation', camera_image[:, :, :3])
            cv2.waitKey(1)

        return self.obs_buf

    def reset_idx(self, env_ids):
        positions = torch.zeros((len(env_ids), self.num_dof), device=self.device)
        velocities = torch.zeros((len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids, :] = positions[:]
        self.dof_vel[env_ids, :] = velocities[:]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions_tensor = 0.99*self.actions_tensor + torch.normal(0.0, 0.05, (1, self.num_envs * self.num_dof), device=self.device)
        #actions_tensor[::self.num_dof] = actions.to(self.device).squeeze() * self.max_push_effort
        vel = gymtorch.unwrap_tensor(self.actions_tensor)
        self.gym.set_dof_velocity_target_tensor(self.sim, vel)

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_cartpole_reward(pole_angle, pole_vel, cart_vel, cart_pos,
                            reset_dist, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # reward is combo of angle deviated from upright, velocity of cart, and velocity of pole moving
    reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)

    # adjust reward for reset agents
    reward = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(reward) * -2.0, reward)
    reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    return reward, reset
