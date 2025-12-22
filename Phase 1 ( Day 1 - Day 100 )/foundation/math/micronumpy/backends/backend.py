_BACKEND = "cpu"

def set_backend(name):
    global _BACKEND
    _BACKEND = name

def get_backend():
    return _BACKEND
