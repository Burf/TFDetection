import inspect

import numpy as np

def dict_function(function = None, keys = []):
    function = [function] if function is not None and not isinstance(function, (tuple, list)) else function
    keys = [keys] if not isinstance(keys, (tuple, list)) else keys
    def wrapper(function):
        function = [function] if np.ndim(function) == 0 else function
        def run(*args, **kwargs):
            args = list(args)
            pre_args = [args.pop(0)] if 0 < len(args) and callable(args[0]) else []
            if 0 < len(args) and isinstance(args[0], dict):
                args_keys = list(args[0].keys())
                new_keys = args_keys + [key for key in keys if key not in args_keys]
                item = args[0].items()
                return_args = False
            else:
                new_keys = keys
                item = zip(keys, args)
                return_args = True
            kwargs = {**{k:v for k, v in item if v is not None}, **kwargs}
            for i, func in enumerate(function):
                if callable(func):
                    func_spec = inspect.getfullargspec(func)
                    remain = {}
                    if func_spec.varargs is None and func_spec.varkw is None:
                        args_keys = func_spec.args + func_spec.kwonlyargs
                        new_kwargs = {}
                        for k, v in kwargs.items():
                            if k in args_keys:
                                new_kwargs[k] = v
                            else:
                                remain[k] = v
                        kwargs = new_kwargs
                    if i == 0:
                        values = func(*pre_args, **kwargs)
                    else:
                        values = func(**kwargs)
                    if not isinstance(values, (tuple, list)):
                        values = (values,)
                    kwargs = {**kwargs, **{k:v for k, v in zip(new_keys, values)}, **remain}
            if return_args:
                result = [kwargs[key] for key in new_keys if key in kwargs]
                result = [r for r in result if r is not None]
                if len(result) == 0:
                    result = None
                elif len(result) == 1:
                    result = result[0]
            else:
                result = {k:v for k, v in kwargs.items() if k in new_keys and v is not None}
            return result
        return run
    if function is not None:
        if 0 < len(function) and callable(function[0]):
            return wrapper(function)
        else:
            keys = function
            function = None
    return wrapper