import tensorflow as tf

from value_based_agent import ValueBasedAgent


class DDQNAgent(ValueBasedAgent):
    """
    Double Deep Q-Network (DDQN) agent.

    Uses the online network to select next actions and
    the target network to evaluate them.
    """

    def __init__(self, name: str, state_size: int, action_size: int, **kwargs):
        # DDQN hyperparameters
        kwargs.setdefault("gamma", 0.99)
        kwargs.setdefault("epsilon", 0.5)
        kwargs.setdefault("epsilon_min", 0.05)
        kwargs.setdefault("epsilon_decay", 0.99)
        kwargs.setdefault("learning_rate", 0.0005)

        super().__init__(name, state_size, action_size, **kwargs)

    @tf.function
    def replay(self, states, actions, rewards, next_states, dones):
        """Train the Q-network using Double DQN Bellman targets."""

        with tf.GradientTape() as tape:
            # Select next actions online and evaluate with target network.
            targets = self.model(states)
            next_actions = tf.argmax(self.model(next_states), axis=1)
            next_q_values = self.target_model(next_states)
            next_q = tf.gather(next_q_values, next_actions, batch_dims=1)

            # Calculate Bellman targets.
            updates = rewards + (1.0 - tf.cast(dones, tf.float32)) * self.gamma * next_q

            # Update targets for the actions taken.
            indices = tf.stack([tf.range(tf.shape(actions)[0]), tf.cast(actions, tf.int32)], axis=1)
            targets_full = tf.tensor_scatter_nd_update(targets, indices, updates)

            # Calculate mean squared error.
            loss = tf.reduce_mean(tf.square(targets_full - self.model(states)))

        # Apply gradients to the online network.
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss