import csv
import os
import json
import pickle
import yaml

import numpy as np

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

def tree_dir(path, file = True, keyword = None, n_skip = 4, n_blank = 4):
    if isinstance(keyword, str):
        keyword = [keyword]
    blank = " " * n_blank
    
    result = []
    for root, dirs, files in os.walk(path):
        level = root.replace(path, "").count(os.sep)
        prefix = blank * level
        sub_prefix = blank * (level + 1)
        result.append("{0}{1}{2}".format(prefix, os.path.basename(root), os.sep))
        if file:
            for i, f in enumerate(files):
                if keyword is not None and not any([key in f for key in keyword]):
                    continue
                if n_skip < 1 or (i + 1) < n_skip:
                    result.append("{0}{1}".format(sub_prefix, f))
                else:
                    result.append("{0}...".format(sub_prefix))
                    break
    return "\n".join(result)

def load_file(path, map = {"\n":""}, mode = "rt"):
    result = []
    with open(path, mode) as file:
        for line in file.readlines():
            if isinstance(map, dict):
                for k, v in map.items():
                    line = line.replace(k, v)
            result.append(line)
    return result

def save_file(data, path, end = "\n", mode = "wt"):
    data = [data] if np.ndim(data) < 1 else data
    with open(path, mode) as file:
        for d in data:
            d = str(d)
            file.write("{0}{1}".format(d, end) if not (end == "" or d[-len(end):] == end) else d)
    return path

def load_csv(path, delimiter = ",", mode = "rt"):
    with open(path, mode) as file:
        result = [line for line in csv.reader(file, delimiter = delimiter)]
    return result

def save_csv(data, path, delimiter = ",", mode = "wt"):
    data = [data] if np.ndim(data) < 1 else data
    data = [data] if np.ndim(data) < 2 else data
    with open(path, mode) as file:
        writer = csv.writer(file, delimiter = delimiter)
        for row in data:
            writer.writerow(row)
    return path

def load_json(path, mode = "rt"):
    with open(path, mode) as file:
        result = json.load(file)
    return result

def save_json(data, path, mode = "wt"):
    with open(path, mode) as file:
        json.dump(data, file)
    return path

def load_yaml(path, mode = "rt"):
    with open(path, mode) as file:
        result = yaml.full_load(file)
    return result

def save_yaml(data, path, mode = "wt"):
    with open(path, mode) as file:
        yaml.dump(data, file)
    return result

def load_pickle(path, mode = "rb"):
    with open(path, mode = mode) as file:
        result = pickle.loads(file.read())
    return result

def save_pickle(data, path , mode = "wb"):
    with open(path, mode = mode) as file:
        file.write(pickle.dumps(data))
    return path