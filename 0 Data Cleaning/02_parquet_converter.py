import os
import pandas as pd

folder_path = 'C:\\Users\\ReinisFals\\OneDrive - Peero, SIA\\Desktop\\stat_arb_research\\stat_arb_research\\data\\A_MERGEABLE\\'

# Get the list of immediate subfolders in the main folder
folders = next(os.walk(folder_path))[1]

for folder in folders:
    subfolder_path = os.path.join(folder_path, folder)
    
    # Get the list of CSV files in the subfolder
    csv_files = [file for file in os.listdir(subfolder_path) if file.endswith('.csv')]
    
    # Convert each CSV file to Parquet
    for csv_file in csv_files:
        csv_file_path = os.path.join(subfolder_path, csv_file)
        output_file = os.path.splitext(csv_file)[0] + '.parquet'
        output_file_path = os.path.join(subfolder_path, output_file)
        
        # Read CSV file using pandas
        df = pd.read_csv(csv_file_path)
        
        # Convert DataFrame to Parquet format
        df.to_parquet(output_file_path)
        
        # Delete the CSV file
        os.remove(csv_file_path)
        
        print(f"Converted {csv_file} to {output_file} and deleted {csv_file}")
    
    print("Done with:", folder)




# def convert_csv_to_parquet(folder_path):
#     # Get the list of immediate subfolders in the main folder
#     folders = next(os.walk(folder_path))[1]

#     for folder in folders:
#         subfolder_path = os.path.join(folder_path, folder)

#         # Get the list of CSV files in the subfolder
#         csv_files = [file for file in os.listdir(subfolder_path) if file.endswith('.csv')]

#         # Convert each CSV file to Parquet
#         for csv_file in csv_files:
#             csv_file_path = os.path.join(subfolder_path, csv_file)
#             output_file = os.path.splitext(csv_file)[0] + '.parquet'
#             output_file_path = os.path.join(subfolder_path, output_file)

#             # Read CSV file using pandas
#             df = pd.read_csv(csv_file_path)

#             # Convert DataFrame to Parquet format
#             df.to_parquet(output_file_path)

#             # Delete the CSV file
#             os.remove(csv_file_path)

#             print(f"Converted {csv_file} to {output_file} and deleted {csv_file}")

#         print("Done with:", folder)

# folder_path = 'C:\\Users\\ReinisFals\\OneDrive - Peero, SIA\\Desktop\\stat_arb_research\\stat_arb_research\\data\\A_MERGEABLE\\'
# convert_csv_to_parquet(folder_path)
