def list_obj(obj):
    if isinstance(obj, list):
        return obj
    else:
        return [obj]