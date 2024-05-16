import cv2
import os

def flash_images(folder_path, duration=100):
    # Get list of image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
    if not image_files:
        print("No image files found in the folder.")
        return
    
    image_files.sort()

    # Create window
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

    # Display each image for a short duration
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        frame = cv2.imread(image_path)

        # Resize image if it's too large
        if frame.shape[0] > 1000 or frame.shape[1] > 1000:
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        cv2.imshow("Image", frame)
        cv2.waitKey(duration)

    cv2.destroyAllWindows()

def create_video_from_images(folder_path, output_video_path, fps=25):
    # Get list of image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
    if not image_files:
        print("No image files found in the folder.")
        return
    
    image_files.sort()

    # Get the dimensions of the first image
    first_image = cv2.imread(os.path.join(folder_path, image_files[0]))
    height, width, _ = first_image.shape

    # Create video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change codec as per your requirement (e.g., 'XVID', 'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write images to video
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        frame = cv2.imread(image_path)
        out.write(frame)

    # Release video writer
    out.release()
    print(f"Video saved as {output_video_path}")

# Example usage
save = False
if save:
    folder_path = "/home/jzhou469/Documents/Scratch/v2_IsaacGymEnvs-Jonathan/isaacgymenvs/generated_data_ballistic_tests/2024-05-15T18:16:51_0"
    output_video_path = "/home/jzhou469/Desktop/example_111.mp4"
    create_video_from_images(folder_path, output_video_path)

# Example usage
view_all = True
if view_all:
    parent_folder = "/home/jzhou469/Documents/Scratch/v2_IsaacGymEnvs-Jonathan/isaacgymenvs/generated_data_ballistic_secondlook"
    folders_list = os.listdir(parent_folder)
    folders_list.sort()
    for folder in folders_list:
        folder_path = os.path.join(parent_folder, folder)
        flash_images(folder_path, duration=40)
else:
    folder_path = "/home/jzhou469/Documents/Scratch/v2_IsaacGymEnvs-Jonathan/isaacgymenvs/generated_data_ballistic_tests/2024-05-15T18:16:51_0"
    flash_images(folder_path, duration=40)