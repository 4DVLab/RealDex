import re
import os




def correct_tf_file_names(path):
    for file in os.listdir(path):
        new_path = path + "/" + file
        try:
            match = re.match(r"(.+) -> (.+)\.txt",file)
            if match:
                link1 = match.group(1)
                link2 = match.group(2)
                new_file_name = link1 + "-" + link2 + ".txt"
                os.rename(new_path,path + "/" + new_file_name)
            elif os.path.isdir(new_path):
                correct_tf_file_names(new_path)
        except PermissionError:
            print(f"permission deied：{new_path}")
        except Exception as e:
            print(f"something goes wrong: {str(e)}")





if __name__ == "__main__":
    correct_tf_file_names("/home/lab4dv/data/")