import os

# Function to create a new folder with the next available number
def create_new_recording_folder(base_folder="captures"):
    # Check if the base folder exists, if not, create it
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    # Find the next available folder name
    n = 1
    while os.path.exists(os.path.join(base_folder, f"recording{n}")):
        n += 1

    # Create the new folder
    new_folder = os.path.join(base_folder, f"recording{n}")
    os.makedirs(new_folder)
    
    return new_folder

# Create the folder and print the path