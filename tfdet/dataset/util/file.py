import os

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