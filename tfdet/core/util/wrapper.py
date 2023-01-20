import functools
import inspect

import numpy as np

def dict_function(function = None, keys = []):
    function = [function] if function is not None and not isinstance(function, (tuple, list)) else function
    keys = [keys] if not isinstance(keys, (tuple, list)) else keys
    def wrapper(function):
        function = [function] if np.ndim(function) == 0 else function
        def run(*args, **kwargs):
            args = list(args)
            map_func = args.pop(0) if 0 < len(args) and (callable(args[0]) or (isinstance(args[0], (tuple, list)) and callable(args[0][0]))) else None #[func] or func or None
            map_func = [map_func] if map_func is not None and not isinstance(map_func, (tuple, list)) else map_func #[func] or None
            return_keys = inspect.getfullargspec((map_func[0] if map_func is not None and 0 < len(map_func) else function[0])).args if len(keys) == 0 else keys
            if 0 < len(args) and isinstance(args[0], dict):
                args_keys = list(args[0].keys())
                return_keys = args_keys + [key for key in return_keys if key not in args_keys]
                item = args[0].items()
                return_args = False
            else:
                item = zip(return_keys, args)
                return_args = True
            args = {k:v for k, v in item if v is not None}
            
            run_func = [functools.partial(function[0], f) for f in map_func] if map_func is not None else function
            base_spec = inspect.getfullargspec(function[0])
            base_keys = base_spec.args + base_spec.kwonlyargs
            for i, func in enumerate(run_func):
                if callable(func):
                    if map_func is not None:
                        func_spec = inspect.getfullargspec(map_func[i])
                        func_keys = func_spec.args + func_spec.kwonlyargs
                        func_keys = func_keys + [k for k in base_keys if k not in func_keys]
                    else:
                        func_spec = inspect.getfullargspec(func)
                        func_keys = func_spec.args + func_spec.kwonlyargs
                    func_kwargs = {k:v for k, v in kwargs.items() if k in func_keys}
                    values = func(**args, **func_kwargs)
                    if not isinstance(values, (tuple, list)):
                        values = (values,)
                    args = {k:v for k, v in zip(return_keys, values)}
            if return_args:
                result = [args[key] for key in return_keys if key in args]
                result = [r for r in result if r is not None]
                if len(result) == 0:
                    result = None
                elif len(result) == 1:
                    result = result[0]
                else:
                    result = tuple(result)
            else:
                result = {k:v for k, v in args.items() if k in return_keys and v is not None}
            return result
        return run
    if function is not None:
        if 0 < len(function) and not isinstance(function[0], str): #and callable(function[0]):
            return wrapper(function)
        else:
            keys = function
            function = None
    return wrapper