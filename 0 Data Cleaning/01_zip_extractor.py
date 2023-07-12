import os
import zipfile


folder_path = 'C:/Users/ReinisFals/OneDrive - Peero, SIA/Desktop/stat_arb_research/stat_arb_research/data/A_MERGEABLE/'

# Get the list of items in the folder
items = os.listdir(folder_path)

# Filter out directories from the list
folders = [item for item in items if os.path.isdir(os.path.join(folder_path, item))]

for i in range(len(folders)):
    folder_path = 'C:/Users/ReinisFals/OneDrive - Peero, SIA/Desktop/stat_arb_research/stat_arb_research/data/A_MERGEABLE/' + str(folders[i])  # Replace with the actual folder path
    output_path = 'C:/Users/ReinisFals/OneDrive - Peero, SIA/Desktop/stat_arb_research/stat_arb_research/data/A_MERGEABLE/' + str(folders[i])  # Replace with the desired output path

    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Extract each file in the folder
    for file in files:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) and file.endswith('.zip'):  # Check if it's a file and a zip archive
            # Perform actions on the file
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(output_path)

            # Remove the ZIP file
            os.remove(file_path)

    print("Done with: " + folders[i])




# def extract_zip_files(folder_path):
#     # Get the list of items in the folder
#     items = os.listdir(folder_path)

#     # Filter out directories from the list
#     folders = [item for item in items if os.path.isdir(os.path.join(folder_path, item))]

#     for i in range(len(folders)):
#         folder = folders[i]
#         folder_path = os.path.join(folder_path, folder)
#         output_path = os.path.join(folder_path, folder)

#         # Get the list of files in the folder
#         files = os.listdir(folder_path)

#         # Extract each file in the folder
#         for file in files:
#             file_path = os.path.join(folder_path, file)
#             if os.path.isfile(file_path) and file.endswith('.zip'):  # Check if it's a file and a zip archive
#                 # Perform actions on the file
#                 with zipfile.ZipFile(file_path, 'r') as zip_ref:
#                     zip_ref.extractall(output_path)

#                 # Remove the ZIP file
#                 os.remove(file_path)

#         print("Done with: " + folder)

# folder_path = 'C:/Users/ReinisFals/OneDrive - Peero, SIA/Desktop/stat_arb_research/stat_arb_research/data/A_MERGEABLE/'
# extract_zip_files(folder_path)