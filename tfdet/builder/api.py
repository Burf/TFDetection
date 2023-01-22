import functools
import inspect

import tfdet
from .util import walk_module, find_module

def build_transform(transform = None, key = "name"):
    """
    transform = {'name':transform name or func, **kwargs} or transform name or func #find module in tfdet.dataset.transform and map kwargs.
                kwargs["sample_size"] > Covnert transform into multi_transform.(If transform doesn't need sample_size.)
    """
    store = walk_module(tfdet.dataset.transform)
    
    result = []
    for info in ([transform] if not isinstance(transform, (tuple, list)) else transform):
        if info is not None:
            if isinstance(info, dict):
                if "name" in info:
                    info = info.copy()
                    func = info.pop("name")
                    if isinstance(func, str):
                        func = find_module(store, func)
                    if callable(func):
                        func_spec = inspect.getfullargspec(func)
                        keys = func_spec.args + func_spec.kwonlyargs
                        if "sample_size" not in keys and "sample_size" in info:
                            func = tfdet.dataset.multi_transform(func, info.pop("sample_size"))
                        func = functools.partial(func, **{k:v for k, v in info.items() if k in keys})
                        result.append(func)
            elif isinstance(info, str):
                result.append(find_module(store, info))
            elif callable(info):
                result.append(info)
    
    if len(result) == 0:
        reuslt = None
    elif len(result) == 1:
        result = result[0]
    return result