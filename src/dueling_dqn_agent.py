import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Add, Subtract, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from base_agent import BaseAgent


class MeanLayer(Layer):
    def __init__(self, axis=None, keepdims=False, **kwargs):
        super(MeanLayer, self).__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return tf.reduce_mean(x, axis=self.axis, keepdims=self.keepdims)

class DuelingDQNAgent(BaseAgent):
    def __init__(self, name, state_size, action_size, **kwargs):
        kwargs.setdefault('gamma', 0.99)
        kwargs.setdefault('epsilon', 0.5)
        kwargs.setdefault('epsilon_min', 0.1)
        kwargs.setdefault('epsilon_decay', 0.995)
        kwargs.setdefault('learning_rate', 0.00025)
        super().__init__(name, state_size, action_size, **kwargs)

    def _build_model(self):
        input_layer = Input(shape=(self.state_size,))
        shared = Dense(64, activation='relu')(input_layer)
        shared = Dense(64, activation='relu')(shared)

        value_stream = Dense(32, activation='relu')(shared)
        value = Dense(1)(value_stream)

        advantage_stream = Dense(32, activation='relu')(shared)
        advantage = Dense(self.action_size)(advantage_stream)

        advantage_mean = MeanLayer(axis=1, keepdims=True)(advantage)
        output = Add()([value, Subtract()([advantage, advantage_mean])])

        model = Model(inputs=input_layer, outputs=output)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    @tf.function
    def replay(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            targets = self.model(states)
            next_q_values = self.target_model(next_states)
            max_next_q = tf.reduce_max(next_q_values, axis=1)

            updates = rewards + (1.0 - tf.cast(dones, tf.float32)) * self.gamma * max_next_q
            indices = tf.stack([tf.range(tf.shape(actions)[0]), tf.cast(actions, tf.int32)], axis=1)
            targets_full = tf.tensor_scatter_nd_update(targets, indices, updates)

            loss = tf.reduce_mean(tf.square(targets_full - self.model(states)))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss
