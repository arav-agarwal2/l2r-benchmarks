from strictyaml import load

def read_config(path, schema):
    """Read configuration file from path
    Args:
        path (str): Relative path to the yaml config file.
    """
    with open(path, "r") as my_file:
        data = my_file.read()
    return load(data, schema).data