import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Add, Subtract, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from value_based_agent import ValueBasedAgent


class MeanLayer(Layer):
    """Calculate the mean along a specified axis."""

    def __init__(self, axis=None, keepdims=False, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return tf.reduce_mean(x, axis=self.axis, keepdims=self.keepdims)


class DuelingDQNAgent(ValueBasedAgent):
    """
    Dueling Deep Q-Network agent.

    Separates state value and action advantage into two streams
    before combining them into Q-values.
    """

    def __init__(self, name: str, state_size: int, action_size: int, **kwargs):
        # Dueling DQN hyperparameters
        kwargs.setdefault("gamma", 0.99)
        kwargs.setdefault("epsilon", 0.5)
        kwargs.setdefault("epsilon_min", 0.1)
        kwargs.setdefault("epsilon_decay", 0.995)
        kwargs.setdefault("learning_rate", 0.00025)

        super().__init__(name, state_size, action_size, **kwargs)

    def _build_model(self):
        """Build the dueling Q-network."""

        # Shared feature layers
        input_layer = Input(shape=(self.state_size,))
        shared = Dense(64, activation="relu")(input_layer)
        shared = Dense(64, activation="relu")(shared)

        # State-value stream
        value_stream = Dense(32, activation="relu")(shared)
        value = Dense(1)(value_stream)

        # Action-advantage stream
        advantage_stream = Dense(32, activation="relu")(shared)
        advantage = Dense(self.action_size)(advantage_stream)

        # Combine state value and centered advantages.
        advantage_mean = MeanLayer(axis=1, keepdims=True)(advantage)
        output = Add()([value, Subtract()([advantage, advantage_mean])])

        model = Model(inputs=input_layer, outputs=output)
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    @tf.function
    def replay(self, states, actions, rewards, next_states, dones):
        """Train the dueling Q-network using Bellman targets."""

        with tf.GradientTape() as tape:
            # Predict current and next-state Q-values.
            targets = self.model(states)
            next_q_values = self.target_model(next_states)
            max_next_q = tf.reduce_max(next_q_values, axis=1)

            # Calculate Bellman targets.
            updates = rewards + (1.0 - tf.cast(dones, tf.float32)) * self.gamma * max_next_q

            # Update targets for the actions taken.
            indices = tf.stack([tf.range(tf.shape(actions)[0]), tf.cast(actions, tf.int32)], axis=1)
            targets_full = tf.tensor_scatter_nd_update(targets, indices, updates)

            # Calculate mean squared error.
            loss = tf.reduce_mean(tf.square(targets_full - self.model(states)))

        # Apply gradients to the online network.
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss