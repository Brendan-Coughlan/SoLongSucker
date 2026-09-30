import tensorflow as tf

from base_agent import BaseAgent


class DQNAgent(BaseAgent):
    """Deep Q-Network agent."""

    def __init__(
        self,
        name: str,
        state_size: int,
        action_size: int,
        **kwargs,
    ) -> None:
        # Default hyperparameters used by the original implementation.
        kwargs.setdefault("gamma", 0.95)
        kwargs.setdefault("epsilon", 0.5)
        kwargs.setdefault("epsilon_min", 0.01)
        kwargs.setdefault("epsilon_decay", 0.995)
        kwargs.setdefault("learning_rate", 0.001)

        super().__init__(name, state_size, action_size, **kwargs)

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

