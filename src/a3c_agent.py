import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from base_agent import BaseAgent
from chip_game_env import GameState

class A3CAgent(BaseAgent):
    def __init__(self, name, state_size, action_size, **kwargs):
        super().__init__(name, state_size, action_size, **kwargs)
        self.gamma = kwargs.get('gamma', 0.99)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.epsilon = kwargs.get('epsilon', 1.0)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.995)
        self.epsilon_min = kwargs.get('epsilon_min', 0.01)
        self.memory = []
        self.actor, self.critic = self._build_actor_critic()
        self.optimizer = Adam(learning_rate=self.learning_rate)

    def _build_actor_critic(self):
        input_state = Input(shape=(self.state_size,))
        dense1 = Dense(64, activation='relu')(input_state)
        dense2 = Dense(64, activation='relu')(dense1)

        actor_output = Dense(self.action_size, activation='softmax')(dense2)
        critic_output = Dense(1)(dense2)

        actor = Model(inputs=input_state, outputs=actor_output)
        critic = Model(inputs=input_state, outputs=critic_output)

        return actor, critic

    def act(self, state, env_state):
        if np.random.rand() <= self.epsilon:
            return self.choose_random_action(env_state)

        policy = self.actor.predict(state, verbose=0)[0]
        if env_state == GameState.CHOOSE_PILE:
            return np.argmax(policy[:6])
        else:
            return np.argmax(policy[6:]) + 6

    def remember(self, state, action, reward, next_state, done, steps):
        self.memory.append((state, action, reward, next_state, done, steps))

    def replay(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            policy = self.actor(states)
            values = self.critic(states)
            next_values = self.critic(next_states)

            advantages = rewards + self.gamma * next_values * (1 - dones) - values

            actor_loss = self._actor_loss(policy, actions, advantages)
            critic_loss = tf.reduce_mean(tf.square(advantages))

            total_loss = actor_loss + critic_loss

        grads = tape.gradient(total_loss, self.actor.trainable_variables + self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables + self.critic.trainable_variables))

    def _actor_loss(self, policy, actions, advantages):
        action_probs = tf.gather(policy, actions, batch_dims=1)
        log_probs = tf.math.log(action_probs)
        return -tf.reduce_mean(log_probs * advantages)

    def update_target_model(self):
        # A3C doesn't use a target model, so this method can be empty
        pass

    def load(self, name):
        self.actor.load_weights(f"{name}_actor")
        self.critic.load_weights(f"{name}_critic")

    def save(self, name):
        self.actor.save_weights(f"{name}_actor")
        self.critic.save_weights(f"{name}_critic")

    def choose_random_action(self, env_state):
        if env_state == GameState.CHOOSE_PILE:
            return np.random.randint(0, 6)  # Assuming 6 piles
        else:
            return np.random.randint(6, self.action_size)
