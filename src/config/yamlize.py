from typing import get_type_hints, TypedDict
import inspect
import strictyaml as sl
import yaml
from enum import Enum
import importlib
import typing as tp


def yamlize(configurable_class):
    # define a new display method

    def convert_type_to_strictyaml(val):

        type_container = tp.get_origin(val)
        if type_container == tuple:
            args = tp.get_args(val)
            arg_list = [convert_type_to_strictyaml(arg) for arg in args]
            return sl.FixedSeq(arg_list)
        elif type_container == list:
            args = tp.get_args(val)
            arg_list = [convert_type_to_strictyaml(arg) for arg in args]
            return sl.Seq(arg_list[0])
        elif type_container is not None:
            raise ValueError(
                f"Type origin {type_container} could not be converted to StrictYAML type. Please add to the convert_type_to_strictyaml function."
            )

        if val == int:
            return sl.Int()
        elif val == str:
            return sl.Str()
        elif val == float:
            return sl.Float()
        elif val == bool:
            return sl.Bool()
        elif val == ConfigurableDict:
            return sl.Map({"name": sl.Str(), "config": sl.Any()})
        else:
            raise ValueError(
                f"Type {val} could not be converted to StrictYAML type. Please add to the convert_type_to_strictyaml function."
            )

    init_types = get_type_hints(configurable_class.__init__)
    init_signature = inspect.signature(configurable_class.__init__)
    init_defaults = {
        k: v.default
        for k, v in init_signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    schema = {}
    for key, val in init_types.items():
        if key == "return":
            continue
        if key in init_defaults:
            key = sl.Optional(key, default=init_defaults[key])
        schema[key] = convert_type_to_strictyaml(val)
    schema = sl.Map(schema)
    configurable_class.schema = schema

    def init_from_config_dict(cls, config):
        config_yamlized = yaml.dump(config)
        config_dict = sl.load(config_yamlized, schema).data
        return cls(**config_dict)

    configurable_class.instantiate_from_config_dict = classmethod(init_from_config_dict)

    def init_from_config(cls, config_file_location):
        with open(config_file_location, "r") as mf:
            yaml_str = mf.read()
        try:
            config_dict = sl.load(yaml_str, schema).data
        except Exception as e:
            raise ValueError(yaml_str, schema, e)
        try:
            return cls(**config_dict)
        except Exception as e:
            import logging

            logging.info(config_dict)
            raise ValueError(config_dict)

    configurable_class.instantiate_from_config = classmethod(init_from_config)

    return configurable_class


class NameToSourcePath(Enum):
    buffer = "src.buffers"
    encoder = "src.encoders"
    logger = "src.loggers"
    runner = "src.runners"
    agent = "src.agents"
    network = "src.networks"


class ConfigurableDict(TypedDict):
    name: str
    config: dict


def create_configurable(config_yaml, name_to_path):
    name_to_path = str(name_to_path.value)
    schema = sl.Map({"name": sl.Str(), "config": sl.Any()})
    with open(config_yaml, "r") as mf:
        yaml_contents = mf.read()
        config_dict = sl.load(yaml_contents, schema).data
    cls = getattr(importlib.import_module(name_to_path), config_dict["name"])

    return cls.instantiate_from_config_dict(config_dict["config"])


def create_configurable_from_dict(config_dict, name_to_path):
    name_to_path = str(name_to_path.value)
    schema = sl.Map({"name": sl.Str(), "config": sl.Any()})
    yaml_contents = yaml.dump(config_dict)
    config_dict = sl.load(yaml_contents, schema).data
    cls = getattr(importlib.import_module(name_to_path), config_dict["name"])
    return cls.instantiate_from_config_dict(config_dict["config"])
