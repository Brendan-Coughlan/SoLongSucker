import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from base_agent import BaseAgent
from chip_game_env import GameState


class ValueBasedAgent(BaseAgent):
    """
    Base class for value-based reinforcement learning agents.

    Provides Q-networks, replay memory, epsilon-greedy action
    selection, and target-network synchronization.
    """

    def __init__(self, name, state_size, action_size, **kwargs):
        super().__init__(name, state_size, action_size, **kwargs)

        # Value-based hyperparameters
        self.gamma = kwargs.get("gamma", 0.95)
        self.epsilon = kwargs.get("epsilon", 1.0)
        self.epsilon_min = kwargs.get("epsilon_min", 0.01)
        self.epsilon_decay = kwargs.get("epsilon_decay", 0.995)

        # Initialize replay memory and Q-networks.
        self.memory = []
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """Build the Q-network."""
        model = Sequential([
            Dense(64, activation="relu", input_shape=(self.state_size,)),
            Dense(64, activation="relu"),
            Dense(self.action_size, activation="linear")
        ])
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """Synchronize the target network."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done, steps):
        """Store an experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done, steps))

    @tf.function
    def act(self, state, env_state):
        """Select an action using epsilon-greedy exploration."""
        if tf.random.uniform(()) <= self.epsilon:
            return self.choose_random_action(env_state)
        else:
            act_values = self.model(state)
            return self.choose_best_action(act_values, env_state)

    def choose_random_action(self, env_state):
        """Select a random action for the current game phase."""
        if env_state == GameState.CHOOSE_PILE:
            return tf.random.uniform((), maxval=6, dtype=tf.int64)
        else:
            return tf.random.uniform((), minval=6, maxval=self.action_size, dtype=tf.int64)

    def choose_best_action(self, act_values, env_state):
        """Select the highest-valued action for the current game phase."""
        if env_state == GameState.CHOOSE_PILE:
            return tf.cast(tf.argmax(act_values[0][:6]), tf.int64)
        else:
            return tf.cast(tf.argmax(act_values[0][6:]) + 6, tf.int64)