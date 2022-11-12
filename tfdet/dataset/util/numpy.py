import numpy as np
import tensorflow as tf

def pad(data, pad_width = None, val = 0):
    if pad_width is not None and 0 < np.max(pad_width):
        data = np.array(data) if not isinstance(data, np.ndarray) else data
        shape = np.shape(data)
        dummy = np.ndim(pad_width) < 2
        pad_width = [pad_width] if dummy else pad_width
        pad_width = list(pad_width) + [[0, 0]] * (len(shape) - len(pad_width))
        new_shape = [s + sum(p) for s, p in zip(shape, pad_width)]
        if val is None:
            pad_data = np.empty(new_shape, dtype = data.dtype)
        else:
            pad_data = np.full(new_shape, val, dtype = data.dtype if np.issubdtype(type(val) if not isinstance(val, np.ndarray) else val.dtype, np.number) else np.object0)
        if np.all([s != 0 for s in shape]):
            region = tuple([slice(None if l == 0 else l, None if r == 0 else -r) for l, r in pad_width])
            pad_data[region if not dummy else region[0]] = data
        data = pad_data
    return data