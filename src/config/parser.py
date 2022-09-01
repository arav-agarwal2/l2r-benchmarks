from strictyaml import load


def read_config(path, schema):
    """Read configuration file from path
    Args:
        path (str): Relative path to the yaml config file.
    """
    with open(path, "r") as my_file:
        data = my_file.read()
    # Load from schema
    data = load(data, schema).data
    # Load first key, as that's just for readability
    key = list(data.keys())[0]
    data = data[key]

    return data
