import os
import re


def replace_hardcoded_paths(root_dir, old_path_fragment, new_path_fragment):
    # Regular expression to find the hardcoded paths
    regex = re.compile(re.escape(old_path_fragment))

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(subdir, file)
            try:
                with open(file_path, 'r') as f:
                    content = f.read()

                # Replace hardcoded paths with the new relative paths
                new_content = regex.sub(new_path_fragment, content)

                # Write the changes back to the file
                with open(file_path, 'w') as f:
                    f.write(new_content)
                print(f"Updated paths in file: {file_path}")
            except Exception as e:
                print(f"Failed to update file {file_path}: {e}")


# Example usage:
# Define the root directory to start the search
root_directory = '/media/jonathan/FastData/objects_and_backgrounds/ABO/centred_objects_urdf'
# Define the fragment of the old path you want to replace
old_path_fragment = '.../centred_objects_gltf/'
# Define the new relative path
new_path_fragment = '/media/jonathan/FastData/objects_and_backgrounds/ABO/centred_objects_gltf/'

replace_hardcoded_paths(root_directory, old_path_fragment, new_path_fragment)