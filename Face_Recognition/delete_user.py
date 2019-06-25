import os
import shutil

folder = 'dataset'
user_num = input("[INPUT] What user would you like to delete? ")
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    split_path = file_path.split(".")
    if split_path[1] == user_num:
        os.unlink(file_path)
