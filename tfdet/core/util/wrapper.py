def dict_function(function = None, extra_keys = []):
    def wrapper(function):
        def run(*args, **kwargs):
            if (0 < len(args) and isinstance(args[0], dict)) or (1 < len(args) and callable(args[0]) and isinstance(args[1], dict)):
                if callable(args[0]):
                    keys = list(args[1].keys())
                    kwargs = {**args[1], **kwargs}
                    args = (args[0],)
                else:
                    keys = list(args[0].keys())
                    kwargs = {**args[0], **kwargs}
                    args = tuple()
                result = function(*args, **kwargs)
                return {k:v for k, v in zip(keys + [key for key in extra_keys if key not in keys], result if isinstance(result, list) or isinstance(result, tuple) else [result])}
            else:
                return function(*args, **kwargs)
        return run
    if callable(function):
        return wrapper(function)
    elif function is not None:
        extra_keys = function
        function = None
    return wrapper