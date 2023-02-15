import tensorflow as tf

def image_to_level(tensor, n_level):
    out = []
    start = 0
    for n in n_level:
        end = start + n
        out.append(tensor[:, start:end])
        start = end
    return out