import os
import json
import requests
import argparse
import gdown

JSON_FILE = 'download/gdown_path.json'

def check_link(url):
    response = requests.head(url, allow_redirects=True)
    if response.status_code == 200:
        print(f"Link is available: {url}")
        return True
    else:
        print(f"Link is not available: {url}. HTTP status code: {response.status_code}")
        return False
    

def download_file(url, dest_folder, filename):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    output = os.path.join(dest_folder, filename)
    gdown.download(url, output, quiet=False)
    print(f"Downloaded {filename} to {dest_folder}")

def main():
    parser = argparse.ArgumentParser(description="Download files listed in a JSON file to a specified folder.")
    parser.add_argument('--dest_folder', type=str, help="The destination folder where files will be downloaded.")
    parser.add_argument('--test', action='store_true', help="Enable test mode to check link availability without downloading.")
    args = parser.parse_args()
    
    if not args.test and not args.dest_folder:
        parser.error("the following argument is required: --dest_folder (unless --test is specified)")

    dest_folder = args.dest_folder
    test_mode = args.test

    # Read the JSON file
    with open(JSON_FILE, 'r') as file:
        files_to_download = json.load(file)
        
    if test_mode:
        failed_paths = {}
        for file_info in files_to_download:
            filename = file_info['filename']
            url = file_info['url']
            if not check_link(url):
                failed_paths[filename] = url
        if len(failed_paths) > 0:
            print("The following files are not available:")
            for filename, url in failed_paths.items():
                print(f"{filename}: {url}")
        else:
            print("All files are available!")
    else:
        for file_info in files_to_download:
            filename = file_info['filename']
            url = file_info['url']
            print(f"Downloading {filename} from {url}...")
            download_file(url, dest_folder, filename)

if __name__ == "__main__":
    main()