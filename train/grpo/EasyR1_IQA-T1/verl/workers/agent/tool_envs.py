import json
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Dict, Tuple


class ToolMeta(ABCMeta):
    """
    Metaclass used to register all concrete tool implementations
    in a global registry on `ToolBase`.
    """

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        # Skip base class itself
        if name == "ToolBase":
            return

        # Require each subclass to define a unique `name`
        if not hasattr(cls, "name"):
            raise AttributeError(f"Tool subclass {name} must define a 'name' attribute.")

        if cls.name in ToolBase.registry:
            existing = ToolBase.registry[cls.name]
            print(
                f"[WARNING] Class {cls.__name__} is trying to register '{cls.name}', "
                f"which has already been registered by {existing.__name__}"
            )
            return

        ToolBase.registry[cls.name] = cls


class ToolBase(ABC, metaclass=ToolMeta):
    """
    Base class for all tools used by the agent environment.

    The design is heavily inspired by the function‑calling / tool‑calling
    conventions used in modern chat models: every tool exposes a JSON‑schema
    description, validates its arguments, and implements a `reset` + `execute`
    interface for multi‑step environments.
    """

    # Global registry: name -> subclass
    registry: Dict[str, "ToolBase"] = {}

    def __init__(self, name: str, description: str = "", parameters: Dict | None = None, **_: Any) -> None:
        self.name = name
        self.description = description
        self.parameters: Dict[str, Any] = parameters or {
            "type": "object",
            "properties": {},
            "required": [],
        }

    # ---------------------------------------------------------------------
    # Factory / Introspection helpers
    # ---------------------------------------------------------------------
    @classmethod
    def create(cls, name: str, description: str = "", parameters: Dict | None = None, **kwargs: Any) -> "ToolBase":
        """
        Instantiate a registered tool by name.

        Example:
            hammer = ToolBase.create("hammer")
        """
        tool_cls = cls.registry.get(name)
        if tool_cls is None:
            raise ValueError(f"No tool registered with name '{name}'")
        return tool_cls(name, description, parameters or {}, **kwargs)

    def get_description(self) -> Dict[str, Any]:
        """
        Return the OpenAI‑style function description for this tool.
        """
        function_desc = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
        return {"type": "function", "function": function_desc}

    def get_json_description(self) -> str:
        """
        Return the JSON‑formatted description of this tool.
        """
        return json.dumps(self.get_description(), indent=2, ensure_ascii=False)

    # ---------------------------------------------------------------------
    # Core tool interface
    # ---------------------------------------------------------------------
    @abstractmethod
    def reset(self, *args: Any, **kwargs: Any) -> None:
        """
        Reset any per‑episode state of the tool/environment.
        """

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """
        Execute one step of the tool.

        Returns:
            tool_result: arbitrary structure describing the observation.
            reward: float, per‑step reward from the tool environment.
            done: whether this episode is finished from the tool's perspective.
            info: debug / logging information.
        """

    def close(self) -> None:
        """
        Optional cleanup hook.

        Concrete tools may override this to release per-episode resources such
        as temporary files or directories. The default implementation is a no-op
        so existing tools remain compatible.
        """
        return None

    # ---------------------------------------------------------------------
    # Argument validation helpers
    # ---------------------------------------------------------------------
    def validate_args(self, args: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate arguments against the JSON‑schema in `self.parameters`.
        """
        if not isinstance(args, dict):
            return False, "Arguments must be a dictionary"

        required = self.parameters.get("required", [])
        for name in required:
            if name not in args:
                return False, f"Missing required parameter: {name}"

        props = self.parameters.get("properties", {})
        for key, value in args.items():
            if key not in props:
                continue
            schema = props[key]
            expected_type = schema.get("type")
            if expected_type and not self._check_type(value, expected_type):
                return False, f"Parameter {key} has incorrect type, should be {expected_type}"

            if "enum" in schema and value not in schema["enum"]:
                enum_values = ", ".join(map(str, schema["enum"]))
                return False, f"Parameter {key} has invalid value, should be one of: {enum_values}"

        return True, "Parameters valid"

    @staticmethod
    def _check_type(value: Any, expected_type: str) -> bool:
        """
        Minimal JSON‑schema type checking.
        """
        if expected_type == "string":
            return isinstance(value, str)
        if expected_type == "number":
            return isinstance(value, (int, float))
        if expected_type == "integer":
            return isinstance(value, int)
        if expected_type == "boolean":
            return isinstance(value, bool)
        if expected_type == "array":
            return isinstance(value, list)
        if expected_type == "object":
            return isinstance(value, dict)
        # For unknown / unsupported types, be lenient.
        return True
