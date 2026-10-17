from abc import abstractmethod

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from base_agent import BaseAgent
from chip_game_env import GameState


class ActorCriticAgent(BaseAgent):
    """
    Abstract base class for actor-critic agents.

    Provides a shared actor-critic network, action selection,
    and common hyperparameters for A2C, A3C, and PPO.
    """

    def __init__(self, name: str, state_size: int, action_size: int, **kwargs):
        super().__init__(name, state_size, action_size, **kwargs)

        # Actor-critic hyperparameters
        self.gamma = kwargs.get("gamma", 0.99)
        self.entropy_coef = kwargs.get("entropy_coef", 0.01)
        self.value_coef = kwargs.get("value_coef", 0.5)

        # Initialize the shared actor-critic network.
        self.model = self._build_model()
        self.optimizer = Adam(learning_rate=self.learning_rate)

    def _build_model(self):
        """Build a shared network with policy and value outputs."""
        inputs = Input(shape=(self.state_size,))
        shared = Dense(64, activation="relu")(inputs)
        shared = Dense(64, activation="relu")(shared)

        policy = Dense(self.action_size, activation="softmax", name="policy")(shared)
        value = Dense(1, name="value")(shared)

        return Model(inputs=inputs, outputs=[policy, value])

    def get_policy_and_value(self, states):
        """Predict action probabilities and state values."""
        return self.model(states)

    def get_action_mask(self, env_state):
        """Return valid actions for the current game phase."""
        if env_state == GameState.CHOOSE_PILE:
            return tf.concat([tf.ones(6), tf.zeros(self.action_size - 6)], axis=0)
        return tf.concat([tf.zeros(6), tf.ones(self.action_size - 6)], axis=0)

    def act(self, state, env_state):
        """Sample an action from the masked policy."""
        policy, _ = self.get_policy_and_value(state)
        mask = self.get_action_mask(env_state)

        masked_policy = policy[0] * mask
        masked_policy = masked_policy / tf.reduce_sum(masked_policy)

        action = tf.random.categorical(tf.math.log(masked_policy[None, :]), 1)
        return tf.cast(action[0, 0], tf.int64)

    @abstractmethod
    def replay(self, *args, **kwargs):
        """Perform an algorithm-specific actor-critic update."""
        pass
