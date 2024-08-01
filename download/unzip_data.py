import os
import zipfile
import argparse

def unzip_files_in_folder(data_folder, dest_folder):
    for item in os.listdir(data_folder):
        if item.endswith('.zip'):
            file_path = os.path.join(data_folder, item)
            print(f"Unzipping {file_path}...")

            extract_folder = os.path.join(dest_folder, os.path.splitext(item)[0])
            if not os.path.exists(extract_folder):
                os.makedirs(extract_folder)

            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)
                
            print(f"Extracted to {extract_folder}")
            
def main():
    parser = argparse.ArgumentParser(description="Unzip files.")
    parser.add_argument('--data_folder', type=str, help="The folder that store all the zipped files.")
    parser.add_argument('--dest_folder', type=str, help="The destination folder that store all the unzipped files.")
    
    args = parser.parse_args()
    
    data_folder = args.data_folder
    dest_folder = args.dest_folder
    
    unzip_files_in_folder(data_folder, dest_folder)
    

if __name__ == "__main__":
    main()