from models.sklearn_models import RandomForest, SupportVector, KNeighbors
from models.neural_models import FeedforwardRegressor, TransformerRegressor
from models.wrappers import SklearnModelWrapper, PytorchModelWrapper


def initialize_model_and_config(train_config, inp_size):
    model_name = train_config["model_config"]["model"]
    model_args = dict(train_config["model_config"])
    model_args.pop("model", None)

    sklearn_models = {
        "RF": RandomForest,
        "SVR": SupportVector,
        "kNN": KNeighbors,
    }

    neural_models = {
        "MLP": FeedforwardRegressor,
        "Transformer": TransformerRegressor,
    }

    config = {k: v for k, v in train_config.items() if k not in ["features", "model_config"]}

    # Merge extra args if available
    if "extra_args" in train_config["model_config"]:
        config.update(train_config["model_config"]["extra_args"])

    # Check and wrap sklearn model
    if model_name in sklearn_models:
        model = sklearn_models[model_name](**model_args)
        config["loss_per_epoch"] = False
        return SklearnModelWrapper(model), config

    # Check and wrap neural model
    elif model_name in neural_models:
        model_args["input_count"] = inp_size
        model = neural_models[model_name](**model_args)
        config.setdefault("epochs", 100)
        config.setdefault("lr", 0.001)
        config["loss_per_epoch"] = config.get("loss_per_epoch", True)
        config["model_type"] = True
        return PytorchModelWrapper(model, config), config

    else:
        raise ValueError(f"Model '{model_name}' not recognized.")
