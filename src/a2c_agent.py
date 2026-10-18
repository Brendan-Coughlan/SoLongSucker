import tensorflow as tf

from actor_critic_agent import ActorCriticAgent


class A2CAgent(ActorCriticAgent):
    """
    Advantage Actor-Critic (A2C) agent.

    Uses one-step advantages to train the actor and critic
    with policy, value, and entropy losses.
    """

    def __init__(self, name, state_size, action_size, **kwargs):
        # A2C hyperparameters
        kwargs.setdefault("gamma", 0.99)
        kwargs.setdefault("learning_rate", 0.0007)
        kwargs.setdefault("entropy_coef", 0.01)
        kwargs.setdefault("value_coef", 0.5)

        super().__init__(name, state_size, action_size, **kwargs)

    @tf.function
    def replay(self, states, actions, rewards, next_states, dones):
        """Train the actor and critic using one-step TD advantages."""

        with tf.GradientTape() as tape:
            # Predict policy probabilities and state values.
            policy, values = self.model(states, training=True)
            _, next_values = self.model(next_states, training=False)

            values = tf.squeeze(values, axis=1)
            next_values = tf.squeeze(next_values, axis=1)

            # Calculate TD targets and advantages.
            targets = rewards + self.gamma * (1.0 - dones) * tf.stop_gradient(next_values)
            advantages = tf.stop_gradient(targets - values)

            # Mask actions according to the game phase.
            choose_pile = states[:, -5]
            pile_mask = tf.concat([tf.ones(6), tf.zeros(self.action_size - 6)], axis=0)
            other_mask = tf.concat([tf.zeros(6), tf.ones(self.action_size - 6)], axis=0)

            masks = choose_pile[:, None] * pile_mask + (1.0 - choose_pile[:, None]) * other_mask
            masked_policy = policy * masks
            masked_policy = masked_policy / tf.reduce_sum(masked_policy, axis=1, keepdims=True)

            # Calculate actor loss.
            indices = tf.stack([tf.range(tf.shape(actions)[0]), tf.cast(actions, tf.int32)], axis=1)
            action_probs = tf.gather_nd(masked_policy, indices)
            log_probs = tf.math.log(action_probs + 1e-8)
            actor_loss = -tf.reduce_mean(log_probs * advantages)

            # Calculate critic loss and policy entropy.
            critic_loss = tf.reduce_mean(tf.square(targets - values))
            entropy = -tf.reduce_sum(masked_policy * tf.math.log(masked_policy + 1e-8), axis=1)
            entropy_loss = tf.reduce_mean(entropy)

            # Combine actor, critic, and entropy losses.
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy_loss

        # Apply gradients to the shared actor-critic network.
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss