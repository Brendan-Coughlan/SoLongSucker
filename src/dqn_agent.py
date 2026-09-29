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
        kwargs.setdefault("epsilon", 1)
        kwargs.setdefault("epsilon_min", 0.01)
        kwargs.setdefault("epsilon_decay", 0.995)
        kwargs.setdefault("learning_rate", 0.001)

        super().__init__(
            name,
            state_size,
            action_size,
            **kwargs,
        )

    @tf.function
    def replay(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        next_states: tf.Tensor,
        dones: tf.Tensor,
    ) -> tf.Tensor:
        """Train the DQN using a batch of experiences."""

        with tf.GradientTape() as tape:
            # Predict Q-values for the current states.
            targets = self.model(states)

            # Estimate the best Q-value for each next state
            # using the target network.
            next_q_values = self.target_model(next_states)

            max_next_q = tf.reduce_max(
                next_q_values,
                axis=1,
            )

            # Calculate the DQN target:
            # reward + gamma * max Q(next_state)
            updates = (
                rewards + (1.0 - tf.cast(dones, tf.float32)) * self.gamma * max_next_q
            )

            # Identify the Q-value corresponding to the
            # action taken in each experience.
            indices = tf.stack(
                [
                    tf.range(tf.shape(actions)[0]),
                    tf.cast(actions, tf.int32),
                ],
                axis=1,
            )

            # Replace those Q-values with their target values.
            targets_full = tf.tensor_scatter_nd_update(
                targets,
                indices,
                updates,
            )

            # Calculate mean squared error.
            predictions = self.model(states)

            loss = tf.reduce_mean(tf.square(targets_full - predictions))

        # Calculate and apply gradients to the main network.
        gradients = tape.gradient(
            loss,
            self.model.trainable_variables,
        )

        self.model.optimizer.apply_gradients(
            zip(
                gradients,
                self.model.trainable_variables,
            )
        )

        return loss
