"""Store schema from older config system. DO NOT USE."""
from strictyaml import (
    load,
    Map,
    Str,
    Int,
    Seq,
    Enum,
    Float,
    Optional,
    Bool,
    CommaSeparated,
)

# Config YAML Schema
cv_trainer_schema = Map(
    {
        "cv_trainer_config": Map(
            {
                "batch_size": Int(),
                "num_epochs": Int(),
                "lr": Float(),
                "model_save_path": Str(),
            }
        )
    }
)


data_fetcher_schema = Map(
    {
        "data_fetcher_config": Map(
            {
                "train_path": Str(),
                "val_path": Str(),
            }
        )
    }
)
