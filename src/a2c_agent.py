import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from base_agent import BaseAgent


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
        if env_state == "choose_pile":
            return np.argmax(policy[:6])
        else:
            return np.argmax(policy[6:]) + 6

    def replay(self, states, actions, rewards, next_states, dones):
        values = self.critic.predict(states, verbose=0)
        next_values = self.critic.predict(next_states, verbose=0)

        advantages = rewards + self.gamma * next_values * (1 - dones) - values

        actor_targets = tf.one_hot(actions, self.action_size)
        self.actor.train_on_batch(states, actor_targets, sample_weight=advantages)
        self.critic.train_on_batch(states, rewards + self.gamma * next_values * (1 - dones))

    def update_target_model(self):
        # A2C doesn't use a target model, so this method can be empty or just pass
        pass

    def load(self, name):
        self.actor.load_weights(f"_actor{name}")
        self.critic.load_weights(f"_critic{name}")

    def save(self, name):
        self.actor.save_weights(f"_actor{name}")
        self.critic.save_weights(f"_critic{name}")

    def choose_random_action(self, env_state):
        if env_state == "choose_pile":
            return np.random.randint(0, 6)  # Assuming 6 rows
        else:
            return np.random.randint(6, self.action_size)
