import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import os

# This function loads BG20K images to GPU memory as tensors
def background_to_GPU(input_directory, sample_size, image_dimensions):
    # Transformations applied to each image
    transform = transforms.Compose([
        transforms.Resize(image_dimensions),  # Resize each image
        transforms.ToTensor()  # Convert image to tensor
    ])

    # Create a dataset from the folder containing images
    dataset = datasets.ImageFolder(root=input_directory, transform=transform)

    # Indices of all images in the dataset
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    # Randomly sample indices
    #sampled_indices = np.random.choice(indices, size=sample_size, replace=False)
    rng = np.random.default_rng(seed=None)
    sampled_indices= rng.choice(indices, size=sample_size, replace=False)

    # Create a DataLoader to load the sampled images
    sampler = SubsetRandomSampler(sampled_indices)
    data_loader = DataLoader(dataset, batch_size=sample_size, sampler=sampler)

    # Assuming CUDA is available, specify the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load sampled images into GPU memory as a tensor
    for images, _ in data_loader:
        images = images.to(device)
        print(images.shape)  # Prints: torch.Size([sample_size, 3, 128, 128])
        # Now `images` is a tensor containing your sampled images on the GPU
    
    return images

# The tensors produced by backgrounds_to_GPU are in the format 
# [batch, color channels, height, width], with 3 colour channels (RGB) that 
# have values between 0 and 1.
# Meanwhile the tensors produced by gym.get_camera_image_gpu_tensor are in the 
# format [height, width, color channels], with 4 colour channels (RGBA) that 
# have values between 0 and 255.
# So, this function formats the backgrounds tensors to match the isaacgym ones.
def match_isaacgym_format(background_image):
    image_hwc = background_image.permute(1, 2, 0)
    #print(image_hwc.shape)
    alpha_values = torch.ones(image_hwc.shape[0], image_hwc.shape[1], 1, device=image_hwc.device)
    image_with_alpha = torch.cat((image_hwc, alpha_values), dim=-1)
    #print(image_with_alpha.shape)
    image_out = image_with_alpha * 255
    return image_out

# This function replaces the background of an isaacgym colour image with
# a BG20K background image, using an isaacgym segmentation image.
def composite_background(cam_color_image, cam_seg_image, backgrounds_tensor, i):
    # Invert segmentation image so pixels are 0 where the object is and 1 everywhere else
    # Create a [128, 128, 4] mask from the [128, 128] segmentation image by 
    # adding a third dimension and copying the data 4 times
    mask = cam_seg_image.unsqueeze(-1).repeat(1, 1, 4)
    inverse_mask = 1 - mask
    # Select a random background image
    #rng = np.random.default_rng(seed=1)
    #background_image_unformatted = backgrounds_tensor[rng.integers(low=0, high=backgrounds_tensor.shape[0]), :, :, :]
    background_image_unformatted = backgrounds_tensor[i, :, :, :]
    background_image = match_isaacgym_format(background_image_unformatted)
    # Replace background of isaacgym colour image
    image_new_background = cam_color_image * mask + background_image * inverse_mask

    return image_new_background

if __name__ == "__main__":
    test = background_to_GPU(input_directory='/imaging/jzhou/stimuli_processing/2024-02-13_15:04', 
                      sample_size=10,
                      image_dimensions=(128,128)
                      )
    
    rng = np.random.default_rng(seed=None)
    background_image = test[rng.integers(low=0, high=test.shape[0]), :, :, :]
    print(background_image.shape)
    print(torch.min(background_image))
    print(torch.max(background_image))
    asdf = match_isaacgym_format(background_image)