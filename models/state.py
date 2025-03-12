from typing import Dict, Any

class StateManager:
    """Simple state manager to share data between agent tools."""
    _instance = None
    _state: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StateManager, cls).__new__(cls)
        return cls._instance

    def set(self, key: str, value: Any):
        """Store a value in state."""
        self._state[key] = value

    def get(self, key: str) -> Any:
        """Get a value from state."""
        return self._state.get(key)

    def has(self, key: str) -> bool:
        """Check if key exists in state."""
        return key in self._state

state = StateManager() 