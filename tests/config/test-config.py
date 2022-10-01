from src.config.parser import read_config
from src.config.schema import (
    experiment_schema,
    encoder_schema,
    agent_schema,
    runner_schema,
    env_schema,
    replay_buffer_schema,
)


def test_env():
    # Test that env.yaml loads properly
    config = read_config("config_files/example_sac/env.yaml", env_schema)
    assert config["track_name"] == "Thruxton"
    assert config["runtime"] == "local"
    assert config["dirhash"] == ""


def test_load():
    # Test that all yamls load properly
    read_config("config_files/example_sac/experiment.yaml", experiment_schema)
    read_config("sconfig_files/example_sac/agent.yaml", agent_schema)
    read_config("config_files/example_sac/buffer.yaml", replay_buffer_schema)
    read_config("config_files/example_sac/encoder.yaml", encoder_schema)
    read_config("config_files/example_sac/env.yaml", env_schema)
    read_config("config_files/example_sac/experiment.yaml", experiment_schema)
    read_config("config_files/example_sac/runner.yaml", runner_schema)
