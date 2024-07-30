import os
import re
import shutil

def clear_cache(data_dir, folder_name, keep_files):
    folder_path = os.path.join(data_dir, folder_name)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename not in keep_files:
            if os.path.isfile(file_path):
                os.remove(file_path)
                # print(f'Deleted {file_path}')
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                # print(f'Directory deleted: {file_path}')

if __name__ == "__main__":
    base_dir = "/storage/group/4dvlab/youzhuo/bags"
    model_name_list = os.listdir(base_dir)
    exclude_list = [] #["body_lotion", "air_duster", "bathroom_cleaner", "beer", "box"]
    file_to_rm = ["merged_pcd", "vis_meshes"]
    
    for model_name in model_name_list:
        if model_name in exclude_list:
            continue
        path = os.path.join(base_dir, model_name)
        for exp_code in os.listdir(path):
            subpath = os.path.join(path, exp_code)
            if os.path.isdir(subpath) and re.match(rf"{model_name}_\d+", exp_code):
                data_dir = os.path.join(base_dir, model_name, exp_code)
                print(data_dir)
                # for folder_name in ["cam0", "cam1", "cam2", "cam3"]:
                #     clear_cache(data_dir,
                #                 folder_name=folder_name,
                #                 keep_files=["rgb", "depth_to_rgb"])
                cache = f"{exp_code.rsplit('_',1)[0]}.txt"
                file_to_rm.append(cache)
                for name in file_to_rm:
                    path_to_rm = os.path.join(data_dir, name)
                    if os.path.exists(path_to_rm):
                        print(path_to_rm)
                        if os.path.isfile(path_to_rm):
                            os.remove(path_to_rm)
                            print(f'Deleted {path_to_rm}')
                        elif os.path.isdir(path_to_rm):
                            shutil.rmtree(path_to_rm)
                            print(f'Deleted dir: {path_to_rm}')
                        

