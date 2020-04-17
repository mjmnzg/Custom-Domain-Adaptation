models = {}

def register_model_fn(name):
    def decorator(fn):
        models[name] = fn
        # set default parameters
        fn.range = None
        fn.mean = None
        fn.bgr = False
        return fn
    return decorator

def get_model_fn(name):
    return models[name]
