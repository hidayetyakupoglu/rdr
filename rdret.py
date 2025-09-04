# ==========================================
# Multi-Agent DQN ET Jamming Simulation with Streamlit Dashboard
# PW & PRI included
# ==========================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_agg import RendererAgg
import threading
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

_lock = threading.Lock()

# -------------------------------
# Realistic Jamming Environment
# -------------------------------
class RealisticJammingEnv:
    def __init__(self, n_jammers=2, n_radars=1):
        self.n_jammers = n_jammers
        self.n_radars = n_radars
        self.agents = ["jammer_" + str(i) for i in range(n_jammers)] + ["radar_" + str(i) for i in range(n_radars)]
        self.radar_params = {
            "freq_min": 8900,
            "freq_max": 9400,
            "PRI_min": 2500,
            "PRI_max": 2531.65,
            "PW_min": 2.3,
            "PW_max": 2.8
        }
        self.action_spaces = {agent: 5 for agent in self.agents if "jammer" in agent}
        self.state = {}
        self.reset()
    
    def reset(self):
        self.state = {}
        for agent in self.agents:
            if "radar" in agent:
                freq = np.random.uniform(self.radar_params["freq_min"], self.radar_params["freq_max"])
                PRI = np.random.uniform(self.radar_params["PRI_min"], self.radar_params["PRI_max"])
                PW = np.random.uniform(self.radar_params["PW_min"], self.radar_params["PW_max"])
                self.state[agent] = np.array([0.0, freq, 5.0, PRI, PW])
            else:
                self.state[agent] = np.zeros(5)
        return self.state
    
    def step(self, actions):
        rewards = {}
        for agent, action in actions.items():
            if "jammer" in agent:
                power = random.uniform(1,5)
                freq_choice = np.random.uniform(self.radar_params["freq_min"], self.radar_params["freq_max"])
                self.state[agent] = np.array([power, freq_choice, 0.0, 0.0, 0.0])
                reward = 0
                for r in [a for a in self.agents if "radar" in a]:
                    radar_freq = self.state[r][1]
                    radar_PRI = self.state[r][3]
                    radar_PW = self.state[r][4]
                    if abs(freq_choice - radar_freq) < 50:
                        impact = 0.1*power*(1 - abs(radar_PW-2.5)/0.5)*(1 - abs(radar_PRI-2515)/31.65)
                        self.state[r][2] = max(0, self.state[r][2]-impact)
                        reward += impact
                rewards[agent] = reward
            else:
                rewards[agent] = self.state[agent][2]
        return self.state, rewards

# -------------------------------
# DQN Agent
# -------------------------------
class DQNAgent:
    def __init__(self, obs_size, n_actions, lr=1e-3, gamma=0.95):
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        
        self.model = nn.Sequential(
            nn.Linear(obs_size,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    
    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.n_actions)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values).item()
    
    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in batch:
            state_t = torch.FloatTensor(state)
            next_state_t = torch.FloatTensor(next_state)
            target = reward + self.gamma * torch.max(self.model(next_state_t))
            output = self.model(state_t)[action]
            loss = self.criterion(output, target.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# -------------------------------
# Streamlit App
# -------------------------------
st.title("Multi-Agent DQN ET Jamming Simulation with PW & PRI")

n_jammers = st.sidebar.number_input("Number of Jammer Agents", 1, 5, 2)
n_radars = st.sidebar.number_input("Number of Radar Agents", 1, 3, 1)
n_episodes = st.sidebar.number_input("Number of Episodes", 50, 500, 200)

env = RealisticJammingEnv(n_jammers=n_jammers, n_radars=n_radars)
jammer_agents = [a for a in env.agents if "jammer" in a]
agents = {agent: DQNAgent(obs_size=5, n_actions=env.action_spaces[agent]) for agent in jammer_agents}

snr_history = {r: [] for r in env.agents if "radar" in r}
jamming_impact_history = []
jammer_freq_history = {j: [] for j in jammer_agents}

st.write("Training and simulation in progress...")
for ep in range(n_episodes):
    state = env.reset()
    episode_reward = 0
    for step in range(10):
        actions = {agent: agents[agent].act(state[agent]) for agent in jammer_agents}
        next_state, rewards = env.step(actions)
        for agent in jammer_agents:
            agents[agent].remember(state[agent], actions[agent], rewards[agent], next_state[agent])
            agents[agent].replay()
        state = next_state
        episode_reward += sum([rewards[a] for a in jammer_agents])
    
    for r in snr_history:
        snr_history[r].append(state[r][2])
    for j in jammer_agents:
        jammer_freq_history[j].append(int(state[j][1]))
    jamming_impact_history.append(episode_reward)
    
    if ep % 25 == 0:
        st.write(f"Episode {ep}: Avg Jamming Impact = {episode_reward:.2f}")

# -------------------------------
# Visualization
# -------------------------------
st.subheader("Radar SNR over Episodes")
fig, ax = plt.subplots()
for r in snr_history:
    ax.plot(snr_history[r], label=r)
ax.set_xlabel("Episode")
ax.set_ylabel("Radar SNR")
ax.legend()
st.pyplot(fig)

st.subheader("Jammer Frequency Heatmap")
jammer_matrix = np.array([jammer_freq_history[j] for j in jammer_agents])
fig2, ax2 = plt.subplots()
sns.heatmap(jammer_matrix, annot=False, fmt="d", cmap="YlOrRd", cbar=True, ax=ax2)
ax2.set_xlabel("Episode")
ax2.set_ylabel("Jammer Agents")
st.pyplot(fig2)

st.subheader("Average Jamming Impact per Episode")
fig3, ax3 = plt.subplots()
ax3.plot(jamming_impact_history, color='red')
ax3.set_xlabel("Episode")
ax3.set_ylabel("Avg Jamming Impact")
st.pyplot(fig3)

st.success("Simulation complete!")
