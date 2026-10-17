from abc import ABC, abstractmethod
from pathlib import Path


class BaseAgent(ABC):
    """
    Abstract base class for reinforcement learning agents.

    Defines common attributes, action selection, training,
    and model persistence interfaces.
    """

    def __init__(self, name: str, state_size: int, action_size: int, **kwargs):
        # Agent attributes
        self.name = name
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.model = None

    @abstractmethod
    def act(self, state, env_state):
        """Select an action based on the current observation."""
        pass

    @abstractmethod
    def replay(self, *args, **kwargs):
        """Perform an algorithm-specific training update."""
        pass

    def save(self, name):
        """Save the model's weights."""
        file_name = f"models/{name}_weights.h5"
        Path(file_name).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(file_name)

    def load(self, name):
        """Load the model's weights."""
        self.model.load_weights(f"models/{name}_weights.h5")