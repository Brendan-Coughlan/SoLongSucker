from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from base_agent import BaseAgent
from chip_game_env import GameState

class PPOAgent(BaseAgent):
    def __init__(self, name, state_size, action_size, **kwargs):
        super().__init__(name, state_size, action_size, **kwargs)
        self.gamma = kwargs.get('gamma', 0.99)
        self.learning_rate = kwargs.get('learning_rate', 0.0003)
        self.epsilon = kwargs.get('epsilon', 0.2)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.995)
        self.epsilon_min = kwargs.get('epsilon_min', 0.01)
        self.clip_ratio = kwargs.get('clip_ratio', 0.2)
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
        old_probs = self.actor.predict(states, verbose=0)
        old_values = self.critic.predict(states, verbose=0)

        advantages = rewards + self.gamma * self.critic.predict(next_states, verbose=0) * (1 - dones) - old_values
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        for _ in range(10):  # Number of optimization epochs
            with tf.GradientTape() as tape:
                new_probs = self.actor(states)
                new_values = self.critic(states)

                ratio = tf.exp(tf.math.log(tf.gather(new_probs, actions, batch_dims=1) + 1e-10) -
                               tf.math.log(tf.gather(old_probs, actions, batch_dims=1) + 1e-10))

                min_advantage = tf.where(
                    advantages > 0,
                    (1 + self.clip_ratio) * advantages,
                    (1 - self.clip_ratio) * advantages,
                )

                actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, min_advantage))
                critic_loss = tf.reduce_mean(tf.square(rewards + self.gamma * self.critic.predict(next_states, verbose=0) * (1 - dones) - new_values))

                total_loss = actor_loss + 0.5 * critic_loss

            grads = tape.gradient(total_loss, self.actor.trainable_variables + self.critic.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables + self.critic.trainable_variables))

    def update_target_model(self):
        # PPO doesn't use a target model, so this method can be empty
        pass

    def load(self, name):
        self.actor.load_weights(f"models/{name}_actor.h5")
        self.critic.load_weights(f"models/{name}_critic.h5")

    def save(self, name):
        Path("models").mkdir(parents=True, exist_ok=True)

        self.actor.save_weights(f"models/{name}_actor.h5")
        self.critic.save_weights(f"models/{name}_critic.h5")

    def choose_random_action(self, env_state):
        if env_state == GameState.CHOOSE_PILE:
            return np.random.randint(0, 6)  # Assuming 6 piles
        else:
            return np.random.randint(6, self.action_size)
