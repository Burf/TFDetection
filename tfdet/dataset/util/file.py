import os
import json

def list_dir(path, keyword = None, absolute = False):
    if isinstance(keyword, str):
        keyword = [keyword]
    
    result = []
    for file in os.listdir(path):
        if keyword is not None and not any([key in file for key in keyword]):
            continue
        
        file_path = os.path.join(path, file)
        if absolute:
            file_path = os.path.abspath(file_path)
        result.append(file_path)
    return result

def walk_dir(path, keyword = None, absolute = False):
    if isinstance(keyword, str):
        keyword = [keyword]
    
    result = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if keyword is not None and not any([key in file for key in keyword]):
                continue
            
            file_path = os.path.join(root, file)
            if absolute:
                file_path = os.path.abspath(file_path)
            result.append(file_path)
    return result

def load_file(path, map = {"\n":""}, mode = "rt"):
    result = []
    with open(path, mode) as file:
        for line in file.readlines():
            if isinstance(map, dict):
                for k, v in map.items():
                    line = line.replace(k, v)
            result.append(line)
    return result

def load_json(path, mode = "rt"):
    with open(path, mode) as file:
        result = json.load(file)
    return result