from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Union, get_args, get_origin, get_type_hints

import yaml


@dataclass(frozen=True)
class Config:
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
    batch_size: int = 64
    learning_rate: float = 1e-3
    latent_dim: int = 128
    beta: float = 1e-3
    epochs: int = 20
    image_size: int = 100
    seed: int = 42
    download: bool = False

    def with_updates(self, **updates: Any) -> Config:
        """Return a new Config instance with updated fields.

        Args:
            **updates: Fields to update in the Config.

        Returns:
            Config: A new Config instance with the specified updates.
        """
        # Build a dict of overrides but ignore any keys explicitly set to None.
        valid_overrides: Dict[str, Any] = {}

        for key, value in updates.items():
            # skip values that are None (treat them as "no override")
            if value is None:
                continue

            valid_overrides[key] = value

        # If no valid overrides, return self (this instance)
        if not valid_overrides:
            return self

        # If there are valid overrides, build a dict of converted values
        # for fields that need conversion
        converted: Dict[str, Any] = {}
        field_types: Dict[str, Any] = get_type_hints(Config)

        for key, value in valid_overrides.items():
            target_type = field_types.get(key)
            converted[key] = self._coerce_value(key, value, target_type)

        # Return a new instance with the updated fields
        return replace(self, **converted)

    def ensure_directories(self) -> None:
        """
        Ensure that the data and output directories exist.
        (If they don't exist, create them.)
        """

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _coerce_value(key: str, value: Any, target_type: Any) -> Any:
        """Coerce a value to the target type if necessary.

        Args:
            key (str): The field name.
            value (Any): The value to coerce.
            target_type (Any): The target type for the field.
        Returns:
            Any: The coerced value.
        """

        if (target_type is None) or (target_type is Any):
            return value

        origin = get_origin(target_type)

        if origin is Union:
            valid_args = [arg for arg in get_args(target_type) if arg is not type(None)]

            if len(valid_args) == 1:
                return Config._coerce_value(key, value, valid_args[0])

        try:
            if target_type is Path:
                return value if isinstance(value, Path) else Path(value)

            if target_type is int:
                return value if isinstance(value, int) else int(value)

            if target_type is float:
                return value if isinstance(value, float) else float(value)

            if target_type is bool:
                if isinstance(value, bool):
                    return value

                if isinstance(value, str):
                    lowered = value.strip().lower()

                    if lowered in {"true", "1", "yes", "y", "on"}:
                        return True

                    if lowered in {"false", "0", "no", "n", "off"}:
                        return False

                return bool(value)

            if origin in {tuple, list}:
                element_types = get_args(target_type)
                element_type = element_types[0] if element_types else Any

                if not isinstance(value, (list, tuple)):
                    raise TypeError(
                        f"Expected a list or tuple for field '{key}', got {type(value)}"
                    )

                # Convert each element
                converted_items = [
                    Config._coerce_value(key, item, element_type) for item in value
                ]

                return tuple(converted_items)

        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Cannot convert value for field '{key}' to type '{target_type}'"
            ) from exc

        # If no special handling needed, return the value as is
        return value


def load_config_from_yaml(
    yaml_path: Union[str, Path], *, overrides: Optional[Dict[str, Any]]
) -> Config:
    """Load a ProjectConfig from YAML with optional runtime overrides.

    Args:
        config_path (Optional[Path], optional): Path to the config YAML file. Defaults to None.
        overrides (Optional[Dict[str, Any]], optional): Overrides for config values. Defaults to None.

    Raises:
        FileNotFoundError: If the config file is not found.
        ValueError: If the config file is invalid.

    Returns:
        Config: The loaded and possibly overridden config.
    """

    # Build a dict of config data from the file and overrides
    config_data: Dict[str, Any] = {}

    # Load from the config file if provided
    if yaml_path:
        # Resolve the path and check it exists
        resolved_path = Path(yaml_path).expanduser()

        # If the file doesn't exist, raise an error
        if not resolved_path.exists():
            raise FileNotFoundError(f"Config file '{resolved_path}' not found.")

        # Load the YAML content
        with resolved_path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}

            # Ensure loaded content is a dict
            if not isinstance(loaded, dict):
                raise ValueError(f"Config file '{resolved_path}' is not a valid YAML mapping.")

            # Update the config data with the loaded content
            config_data.update(loaded)

    if overrides:
        config_data.update({k: v for k, v in overrides.items() if v is not None})

    base = Config()

    return base.with_updates(**config_data)
