from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from base_agent import BaseAgent
from chip_game_env import GameState


class A2CAgent(BaseAgent):
    def __init__(self, name, state_size, action_size, **kwargs):
        super().__init__(name, state_size, action_size, **kwargs)
        self.actor, self.critic = self._build_actor_critic()

    def _build_actor_critic(self):
        input_state = Input(shape=(self.state_size,))
        dense1 = Dense(16, activation='relu')(input_state)
        dense2 = Dense(16, activation='relu')(dense1)

        actor_output = Dense(self.action_size, activation='softmax')(dense2)
        critic_output = Dense(1)(dense2)

        actor = Model(inputs=input_state, outputs=actor_output)
        critic = Model(inputs=input_state, outputs=critic_output)

        actor.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='categorical_crossentropy')
        critic.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')

        return actor, critic

    def act(self, state, env_state):
        if np.random.rand() <= self.epsilon:
            return self.choose_random_action(env_state)

        policy = self.actor.predict(state, verbose=0)[0]
        if env_state == GameState.CHOOSE_PILE:
            return np.argmax(policy[:6])
        else:
            return np.argmax(policy[6:]) + 6

    def replay(self, states, actions, rewards, next_states, dones):
        # Convert everything to predictable NumPy shapes/types
        states = np.asarray(states, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)

        actions = np.asarray(actions, dtype=np.int32).reshape(-1)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        dones = np.asarray(dones, dtype=np.float32).reshape(-1)

        # Critic outputs (batch_size, 1), so flatten to (batch_size,)
        values = self.critic.predict(states, verbose=0).reshape(-1)
        next_values = self.critic.predict(
            next_states,
            verbose=0
        ).reshape(-1)

        # TD target:
        # reward + discounted value of next state unless terminal
        td_targets = (
            rewards
            + self.gamma * next_values * (1.0 - dones)
        )

        # A(s,a) = TD target - V(s)
        advantages = td_targets - values

        # One-hot encode selected actions
        actor_targets = tf.one_hot(
            actions,
            depth=self.action_size
        )

        # Train policy using advantage as sample weight
        self.actor.train_on_batch(
            states,
            actor_targets,
            sample_weight=advantages
        )

        # Critic learns the TD target
        self.critic.train_on_batch(
            states,
            td_targets
        )

    def update_target_model(self):
        # A2C doesn't use a target model, so this method can be empty or just pass
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
