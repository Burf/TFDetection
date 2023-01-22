import types
def walk_module(module, depth = 0, items = [types.FunctionType, type]): #type > class type
    store = {}
    for name in dir(module):
        attr = getattr(module, name)
        if isinstance(attr, types.ModuleType):
            if 0 < depth:
                r = walk_modules(attr, depth - 1, items = items)
                store.update(r)
        else:
            for item in items:
                if isinstance(attr, item):
                    store["{0}.{1}".format(module.__name__, name)] = attr
                    break
    return store

def find_module(store, name):
    result = None
    for k, v in store.items():
        if name == k[-len(name):]:
            result = v
            break
    return result