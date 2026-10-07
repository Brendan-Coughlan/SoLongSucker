from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Add, Subtract, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import numpy as np
import random
import matplotlib.pyplot as plt
from chip_game_env import GameState

class BaseAgent:
    def __init__(self, name, state_size, action_size, **kwargs):
        self.name = name
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.memory = []
        self.gamma = kwargs.get('gamma', 0.95)
        self.epsilon = kwargs.get('epsilon', 1.0)
        self.epsilon_min = kwargs.get('epsilon_min', 0.01)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.995)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(self.state_size,)),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done, steps):
        self.memory.append((state, action, reward, next_state, done, steps))

    @tf.function
    def act(self, state, env_state):
        if tf.random.uniform(()) <= self.epsilon:
            return self.choose_random_action(env_state)
        else:
            act_values = self.model(state)
            return self.choose_best_action(act_values, env_state)

    def choose_random_action(self, env_state):
        if env_state == GameState.CHOOSE_PILE:
            return tf.random.uniform((), maxval=6, dtype=tf.int64)  # Assuming 6 rows
        else:
            return tf.random.uniform((), minval=6, maxval=self.action_size, dtype=tf.int64)

    def choose_best_action(self, act_values, env_state):
        if env_state == GameState.CHOOSE_PILE:
            return tf.cast(tf.argmax(act_values[0][:6]), tf.int64)  # Assuming 6 rows
        else:
            return tf.cast(tf.argmax(act_values[0][6:]) + 6, tf.int64)

    def replay(self, batch_size):
        # This method will be implemented in child classes
        pass

    def load(self, name):
        self.model.load_weights(f"models/{name}_weights.h5")

    def save(self, name):
        file_name = f"models/{name}_weights.h5"
        Path(file_name).parent.mkdir(parents=True, exist_ok=True)

        self.model.save_weights(file_name)