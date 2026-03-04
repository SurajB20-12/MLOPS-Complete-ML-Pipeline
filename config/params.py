import yaml
from config.logger import get_logger

logger = get_logger("params")


class ParamsManager:
    def __init__(self, params_path: str = "params.yaml"):
        self.params = self._load_params(params_path)

    def _load_params(self, params_path: str) -> dict:
        try:
            with open(params_path, "r") as file:
                params = yaml.safe_load(file)
            logger.debug("Parameters loaded from %s", params_path)
            return params
        except Exception as e:
            logger.error("Failed to load params: %s", e)
            raise

    def get(self, *keys):
        """
        Access nested keys safely.
        Example:
        config.get("model_building", "n_estimators")
        """
        value = self.params
        for key in keys:
            value = value.get(key)
            if value is None:
                raise KeyError(f"Key {' -> '.join(keys)} not found in params.yaml")
        return value
