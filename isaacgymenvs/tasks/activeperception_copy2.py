import numpy as np
import os
import torch
import datetime

from isaacgym import gymutil, gymtorch, gymapi
from .base.vec_task import VecTask

import cv2  # importing for visualization
#import isaacgymenvs.tasks.utils.activeperception_utils as utils

from . import background_tools
from . import my_config
import time
import sys
import imageio

def conditional_print(*args):
    if not my_config.SILENT:
        print(*args)
    else:
        pass

import xml.etree.ElementTree as ET

def modify_urdf(file_name):
    # Load the URDF file
    tree = ET.parse(file_name)
    root = tree.getroot()

    # Generate a random vector using NumPy and normalize it
    random_vector = np.random.rand(3)
    normalized_vector = random_vector / np.linalg.norm(random_vector)

    # Collect joints to remove
    joints_to_remove = ['y_rotation_joint', 'z_rotation_joint']
    links_to_remove = ['x_rotating_link', 'y_rotating_link']

    # Remove specified joints
    for joint in root.findall('joint'):
        if joint.get('name') in joints_to_remove:
            root.remove(joint)

    # Remove specified links
    for link in root.findall('link'):
        if link.get('name') in links_to_remove:
            root.remove(link)

    # Update the x_rotation_joint to connect base_link directly to rectangle_link
    for joint in root.findall('joint'):
        if joint.get('name') == 'x_rotation_joint':
            joint.find('child').set('link', 'rectangle_link')
            axis = joint.find('axis')
            axis.set('xyz', f"{normalized_vector[0]} {normalized_vector[1]} {normalized_vector[2]}")

    # Save the modified tree to a new file
    new_file_name = file_name
    tree.write(new_file_name)


class activeperception(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        conditional_print('running __init__... instance of activeperception created')

        self.cfg = cfg

        self.reset_dist = self.cfg["env"]["resetDist"]

        self.max_push_effort = self.cfg["env"]["maxEffort"]
        self.max_episode_length = 500

        self.cfg["env"]["numObservations"] = 2
        self.cfg["env"]["numActions"] = 1

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        #self.actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)
        self.torch_rng = torch.Generator(device=self.device)
        self.torch_rng.seed()
        #test = torch.normal(0.0, 1.0, (1, self.num_envs * self.num_dof), generator=self.torch_rng, device=self.device)
        #vel = generate_random_joint_velocities(self.num_envs, 1.)
        #vel = torch.tensor(np.expand_dims(vel.flatten(), axis=0), dtype=torch.float32).to(self.device)
        #vel = torch.tensor(vel, dtype=torch.float32).to(self.device)
        self.actions_tensor = torch.ones(size=(1, self.num_envs * self.num_dof), device=self.device)

        self.gym.viewer_camera_look_at(
            self.viewer, None, gymapi.Vec3(15.0, 15.0, 15.0), gymapi.Vec3(0, 0, 0))
        
        # this is a bit of a janky way to count the number of times compute_observations is called
        self.compute_observations_count = 0

        # create a buffer to store velocity metadata
        self.velocity_sequences = []

        # save start time to customize output folder names
        current_datetime = datetime.datetime.now()
        self.start_time = current_datetime.strftime('%Y-%m-%dT%H:%M:%S')

        # start "stopwatch"
        self.stopwatch_start = time.time()

    def create_sim(self):
        conditional_print('running create_sim')

        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis = self.cfg["sim"]["up_axis"]
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        #self._create_ground_plane()  # no need for a ground plane
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        
        # load background images into GPU (don't really know if I should put this here, but this function is usually only called once at the beginning, so it seems appropriate enough?)
        self.bkgs = background_tools.background_to_GPU(input_directory='/media/jonathan/FastData/objects_and_backgrounds/backgrounds/',
                      sample_size=self.num_envs,
                      image_dimensions=(256,256)
                      )
        # create directory for saving background swapped images
        self.img_dir = my_config.OUTPUT_LOCATION
        if not os.path.exists(self.img_dir):
            os.mkdir(self.img_dir)

    def _create_ground_plane(self):
        conditional_print('running _create_ground_plane')
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)        

    def _create_envs(self, num_envs, spacing, num_per_row):
        conditional_print('running _create_envs')

        # define plane on which environments are initialized
        lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # set asset root and instantiate asset options
        #TEMP2 asset_root = "assets_temp"
        #TEMP3 asset_root = "glb_write_rough_assets" #TEMP2
        #TEMP4 asset_root = "/imaging/jzhou/stimuli_processing/module_output_tests/centre_objects_test/centred_objects_urdf" #TEMP3
        asset_root = "/media/jonathan/FastData/objects_and_backgrounds/ABO/centred_objects_urdf" #TEMP4
        asset_list = os.listdir(asset_root) #TEMP4
        #TEMP4 asset_metadata = "../../ABO_exploration/ABO_size_filtered_strict.csv"
        asset_metadata = "/media/jonathan/FastData/objects_and_backgrounds/ABO/metadata.csv" #TEMP4
        #asset_paths = utils.get_column(asset_metadata, 'path')

        # initialize random generator (used to select assets in environment creation loop)
        # set seed=None for different choices each time train.py is run
        rng = np.random.default_rng(seed=None)

        asset_options = gymapi.AssetOptions()
        asset_options.use_mesh_materials = True
        asset_options.fix_base_link = True

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

        self.object_handles = []
        self.camera_handles = []
        self.envs = []

        self.cam_tensors_color = []
        self.cam_tensors_seg = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            #TEMP asset_file = utils.make_urdf(rng.choice(asset_paths, replace=False))
            #TEMP4 asset_file = os.listdir(asset_root)[i] #TEMP
            asset_file = rng.choice(asset_list, replace=False) #TEMP4
            #current_datetime = datetime.datetime.now()
            #asset_name = os.path.splitext(asset_file)[0]
            #dir_name = asset_name + current_datetime.strftime('_%Y-%m-%d_%H:%M')
            #dir_path = self.img_dir + '/' + dir_name
            #try:
                #os.mkdir(dir_path)
            #except FileExistsError:
                #dir_path = asset_name + current_datetime.strftime('_%Y-%m-%dT%H:%M:%S')
                #dir_path = self.img_dir + '/' + dir_name
                #os.mkdir(dir_path)

            # Example usage
            modify_urdf(os.path.join(asset_root, asset_file))
            #asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
            asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
            self.num_dof = self.gym.get_asset_dof_count(asset)

            actor_handle = self.gym.create_actor(env_ptr, asset, pose, "object", i, 1)

            props = self.gym.get_actor_dof_properties(env_ptr, actor_handle)
            props["driveMode"].fill(gymapi.DOF_MODE_VEL)  # set control mode as velocity control
            props["stiffness"].fill(0.0)
            props["damping"].fill(1.0)
            self.gym.set_actor_dof_properties(env_ptr, actor_handle, props)

            # set rigid body segmentation ID
            num_bodies = self.gym.get_actor_rigid_body_count(env_ptr, actor_handle)
            conditional_print(f'num_bodies: {num_bodies}')
            self.gym.set_rigid_body_segmentation_id(env_ptr, actor_handle, 1, 1)

            # add camera sensors to capture the scene
            camera_props = gymapi.CameraProperties()
            camera_props.width = 256
            camera_props.height = 256
            camera_props.horizontal_fov = 60
            camera_props.enable_tensors = True  # we'll enable this when the images are passed directly to a neural network on the GPU
            camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)
            pos = np.random.rand(3)  # randomize camera position
            pos = pos / np.linalg.norm(pos) * 0.381  # scale camera position so that it lies on a sphere
            self.gym.set_camera_location(camera_handle, env_ptr, gymapi.Vec3(pos[0], pos[1], pos[2]), gymapi.Vec3(0, 0, 0))

            # obtain color image camera tensor
            cam_tensor_color = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR)
            conditional_print("Got color image camera tensor with shape", cam_tensor_color.shape)

            # wrap color image camera tensor in a pytorch tensor
            torch_cam_tensor_color = gymtorch.wrap_tensor(cam_tensor_color)
            self.cam_tensors_color.append(torch_cam_tensor_color)
            conditional_print("  Torch color image camera tensor device:", torch_cam_tensor_color.device)
            conditional_print("  Torch color image camera tensor shape:", torch_cam_tensor_color.shape)

            # obtain segmentation image camera tensor
            cam_tensor_seg = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_SEGMENTATION)
            conditional_print("Got segmentation image camera tensor with shape", cam_tensor_seg.shape)

            # wrap segmentation image camera tensor in a pytorch tensor
            torch_cam_tensor_seg = gymtorch.wrap_tensor(cam_tensor_seg)
            self.cam_tensors_seg.append(torch_cam_tensor_seg)
            conditional_print("  Torch segmentation image camera tensor device:", torch_cam_tensor_seg.device)
            conditional_print("  Torch segmentation camera tensor shape:", torch_cam_tensor_seg.shape)

            self.envs.append(env_ptr)
            self.object_handles.append(actor_handle)
            self.camera_handles.append(camera_handle)

    def compute_reward(self):
        conditional_print('running compute_reward')

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
        conditional_print('running compute_observations')

        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)

        self.obs_buf[env_ids, 0] = self.dof_pos[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 1] = self.dof_vel[env_ids, 0].squeeze()

        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        for i in env_ids:
            # Use camera tensor when you want to keep the image on the GPU to pass it to a model
            ##camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_COLOR)
            if False:
                print(f'compute_observations_count: {self.compute_observations_count}')
            if self.compute_observations_count < my_config.NUM_FRAMES:
                #fname_color = os.path.join(self.img_dir, "cam_color-%04d-%04d.png" % (self.compute_observations_count, i))
                #cam_img_color = self.cam_tensors_color[i].cpu().numpy()
                #imageio.imwrite(fname_color, cam_img_color)

                #fname_seg = os.path.join(self.img_dir, "cam_seg-%04d-%04d.png" % (self.compute_observations_count, i))
                #cam_img_seg = self.cam_tensors_seg[i].cpu().numpy()
                #cam_img_seg_visualized = cam_img_seg * 255
                #imageio.imwrite(fname_seg, cam_img_seg_visualized.astype(np.uint8))

                sequence_dirname = self.start_time + '_' + str(i)
                sequence_path = os.path.join(self.img_dir, sequence_dirname)
                if not os.path.exists(sequence_path):
                    os.mkdir(sequence_path)
                fname_composite = os.path.join(sequence_path, "composite-%04d-%04d.png" % (i, self.compute_observations_count))
                composite_img = background_tools.composite_background(cam_color_image=self.cam_tensors_color[i], 
                                                                      cam_seg_image=self.cam_tensors_seg[i], 
                                                                      backgrounds_tensor=self.bkgs,
                                                                      i=i
                                                                      )
                composite_img_out = composite_img.cpu().numpy()
                imageio.imwrite(fname_composite, composite_img_out.astype(np.uint8))

                if self.compute_observations_count == 1:
                    save_background_path = os.path.join(sequence_path, "background.png")
                    save_background = background_tools.match_isaacgym_format(self.bkgs[i, :, :, :])
                    save_background_out = save_background.cpu().numpy()
                    imageio.imwrite(save_background_path, save_background_out.astype(np.uint8))

            #camera_image = self.gym.get_camera_image(self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_COLOR)
            #camera_image = camera_image.reshape((camera_image.shape[0], camera_image.shape[0], 4))
            #camera_image = camera_image[:, :, :3]
            #camera_image = cv2.cvtColor(camera_image, cv2.COLOR_RGB2BGR)
            #cv2.imshow('Camera Observation', camera_image[:, :, :3])
            #cv2.waitKey(1)
        
        if self.compute_observations_count == my_config.NUM_FRAMES:
            self.stopwatch_end = time.time()
            self.elapsed_time = self.stopwatch_end - self.stopwatch_start
            print("Elapsed time:", self.elapsed_time, "seconds")
            print("done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done done")
            sys.exit()
            #gym.destroy_viewer(self.viewer)
            #gym.destroy_sim(self.sim)

        self.gym.end_access_image_tensors(self.sim)
        self.compute_observations_count += 1

        return self.obs_buf

    def reset_idx(self, env_ids):
        conditional_print('running reset_idx')

        positions = torch.zeros((len(env_ids), self.num_dof), device=self.device)
        velocities = torch.zeros((len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids, :] = positions[:]
        self.dof_vel[env_ids, :] = velocities[:]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)
        #self.actions_tensor = torch.normal(0.0, 1.0, (1, self.num_envs * self.num_dof), device=self.device)

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        conditional_print('running pre_physics_step')
        
        #if self.compute_observations_count == 1:
            #self.actions_tensor = 0.99*self.actions_tensor + torch.normal(0.0, 1.0, (1, self.num_envs * self.num_dof), device=self.device)
            #vel = gymtorch.unwrap_tensor(self.actions_tensor)
            #self.gym.set_dof_velocity_target_tensor(self.sim, vel)
            #print("kick")
        
        #else:
            #self.actions_tensor = 0.99*self.actions_tensor + torch.normal(0.0, 0.0001, (1, self.num_envs * self.num_dof), device=self.device)
            #vel = gymtorch.unwrap_tensor(self.actions_tensor)
            #self.gym.set_dof_velocity_target_tensor(self.sim, vel)

        if self.compute_observations_count == 0:
            vel = gymtorch.unwrap_tensor(self.actions_tensor)
            self.gym.set_dof_velocity_target_tensor(self.sim, vel)
        
        conditional_print(self.actions_tensor.cpu().numpy())

        #forces = gymtorch.unwrap_tensor(self.actions_tensor)
        #self.gym.set_dof_actuation_force_tensor(self.sim, forces)

        #actions_tensor[::self.num_dof] = actions.to(self.device).squeeze() * self.max_push_effort

    def post_physics_step(self):
        conditional_print('running post_physics_step')

        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        #if len(env_ids) > 0:
            #self.reset_idx(env_ids)

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
