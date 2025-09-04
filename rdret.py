# -*- coding: utf-8 -*-
# ==========================================
# Streamlit Multi-Agent DQN ET Jamming Dashboard with Success Scores
# ==========================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# -------------------------------
# Multi-Agent Jamming Ortamı
# -------------------------------
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import AgentSelector
from gymnasium import spaces

class RealisticJammingEnv(AECEnv):
    def __init__(self, n_jammers=2, n_radars=1, max_power=10, n_freq=5):
        super().__init__()
        self.n_jammers = n_jammers
        self.n_radars = n_radars
        self.agents = ["jammer_" + str(i) for i in range(n_jammers)] + ["radar_" + str(i) for i in range(n_radars)]
        self.possible_agents = self.agents[:]
        self.max_power = max_power
        self.n_freq = n_freq

        # Her ajan için action space: power x freq kombinasyonu
        self.action_spaces = {agent: spaces.Discrete(max_power * n_freq) for agent in self.agents}

        # Observation: [Power, Freq, SNR, PW, PRI]
        self.observation_spaces = {agent: spaces.Box(low=0, high=max_power, shape=(5,), dtype=np.float32)
                                   for agent in self.agents}

    def reset(self, seed=None, options=None):
        # State: [current_power, current_freq, SNR, PW, PRI]
        self.state = {
            agent: np.array([0.0, np.random.randint(self.n_freq), np.random.rand()*5,
                             np.random.uniform(2.0, 3.0), np.random.uniform(2500.0, 2532.0)], dtype=np.float32)
            for agent in self.agents
        }
        self.agent_selection = AgentSelector(self.agents)
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent:{} for agent in self.agents}
        return self.state

    def observe(self, agent):
        return self.state[agent]

    def step(self, action_dict):
        rewards = {}
        for agent, action in action_dict.items():
            power = action // self.n_freq
            freq = action % self.n_freq

            self.state[agent][:2] = [power, freq]  # Power ve frekans

            reward = 0
            if "jammer" in agent:
                # Jammer radarın SNR'ını düşürür
                for r in [a for a in self.agents if "radar" in a]:
                    # Frekans, PW ve PRI etkisi
                    freq_match = freq == int(self.state[r][1])
                    pw_factor = np.clip(self.state[r][3] / 3.0, 0.5, 1.0)
                    pri_factor = np.clip(2500.0 / self.state[r][4], 0.5, 1.0)
                    reduction = 0.1 * power * freq_match * pw_factor * pri_factor
                    self.state[r][2] = max(0, self.state[r][2] - reduction)
                reward = np.sum([5 - self.state[r][2] for r in self.agents if "radar" in r])
            else:
                reward = self.state[agent][2]  # Radar ödülü

            self._cumulative_rewards[agent] += reward
            rewards[agent] = reward
        return self.state, rewards

# -------------------------------
# DQN Ajanı
# -------------------------------
class DQNAgent:
    def __init__(self, obs_size, n_actions, lr=1e-3, gamma=0.95):
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=2000)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions-1)
        state_t = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            return int(torch.argmax(self.model(state_t)).item())

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in batch:
            state_t = torch.FloatTensor(state).to(self.device)
            next_state_t = torch.FloatTensor(next_state).to(self.device)
            reward_t = torch.tensor(reward, dtype=torch.float32).to(self.device)
            target = reward_t + self.gamma * torch.max(self.model(next_state_t))
            current = self.model(state_t)[action]
            loss = self.loss_fn(current, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# -------------------------------
# Streamlit Başlık ve Parametreler
# -------------------------------
st.set_page_config(page_title="Multi-Agent ET Jamming", layout="wide")
st.title("Multi-Agent DQN ET Jamming Simulation with Success Scores")

n_episodes = st.sidebar.slider("Number of Episodes", 50, 500, 200, 25)
n_jammers = st.sidebar.slider("Number of Jammers", 1, 5, 2)
n_radars = st.sidebar.slider("Number of Radars", 1, 3, 1)
max_power = st.sidebar.slider("Max Jammer Power", 1, 20, 10)
n_freq = st.sidebar.slider("Number of Frequencies", 2, 10, 5)

st.sidebar.write("Click 'Start Simulation' to run.")
start_button = st.sidebar.button("Start Simulation")

# -------------------------------
# Simülasyon Başlat
# -------------------------------
if start_button:
    env = RealisticJammingEnv(n_jammers=n_jammers, n_radars=n_radars, max_power=max_power, n_freq=n_freq)
    agents = {agent: DQNAgent(obs_size=5, n_actions=env.action_spaces[agent].n) for agent in env.agents}

    snr_baseline = 5.0
    snr_history = {r: [] for r in env.agents if "radar" in r}
    jammer_freq_history = {j: [] for j in env.agents if "jammer" in j}
    detection_scores = {r: [] for r in snr_history}
    jamming_impact_scores = []
    state_history = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for ep in range(n_episodes):
        state = env.reset()
        action_dict = {agent: agents[agent].act(state[agent]) for agent in env.agents}
        next_state, rewards = env.step(action_dict)

        # Hafızaya ekle ve replay
        for agent in env.agents:
            agents[agent].remember(state[agent], action_dict[agent], rewards[agent], next_state[agent])
            agents[agent].replay()

        state_history.append(next_state)
        for r in snr_history:
            snr = next_state[r][2]
            snr_history[r].append(snr)
            detection_scores[r].append(snr / snr_baseline)
        for j in jammer_freq_history:
            jammer_freq_history[j].append(int(next_state[j][1]))

        jamming_impact = np.mean([snr_baseline - next_state[r][2] for r in snr_history])
        jamming_impact_scores.append(jamming_impact)

        progress_bar.progress((ep + 1)/n_episodes)
        status_text.text(f"Episode {ep+1}/{n_episodes}: Avg Jamming Impact = {jamming_impact:.2f}")
        time.sleep(0.01)

    st.success("Simulation Complete!")

    # -------------------------------
    # Başarı Skorları Tablosu
    # -------------------------------
    st.subheader("Agent Success Scores")
    success_table = []
    for agent in env.agents:
        if "radar" in agent:
            avg_score = np.mean(detection_scores[agent])
            success_table.append([agent, "Radar", f"{avg_score:.2f}", f"{max(detection_scores[agent]):.2f}", f"{min(detection_scores[agent]):.2f}"])
        else:
            avg_score = np.mean([snr_baseline - next_state[r][2] for r in snr_history])
            success_table.append([agent, "Jammer", f"{avg_score:.2f}", "-", "-"])

    st.table(success_table)

    # -------------------------------
    # Dashboard Grafikleri
    # -------------------------------
    fig, axes = plt.subplots(2,2, figsize=(14,10))
    radars = list(snr_history.keys())
    jammers = list(jammer_freq_history.keys())

    # Radar SNR
    axes[0,0].set_title("Radar SNR at Final Episode")
    axes[0,0].bar(radars, [state[r][2] for r in radars], color='skyblue')
    axes[0,0].set_ylim(0,5)

    # Radar Detection Score
    axes[0,1].set_title("Radar Detection Score")
    for r in radars:
        axes[0,1].plot(detection_scores[r], label=r)
    axes[0,1].set_xlabel("Episode")
    axes[0,1].set_ylabel("Detection Score")
    axes[0,1].legend()

    # Jammer Frequency Heatmap
    axes[1,0].set_title("Jammer Frequency Choices")
    jammer_matrix = np.array([jammer_freq_history[j] for j in jammers])
    sns.heatmap(jammer_matrix, annot=False, fmt="d", cmap="YlOrRd", cbar=True, ax=axes[1,0])
    axes[1,0].set_xlabel("Episode")
    axes[1,0].set_ylabel("Jammer Agents")

    # Jammer Impact Score
    axes[1,1].set_title("Jammer Impact Score")
    axes[1,1].plot(jamming_impact_scores, color='red', label='Jamming Impact')
    axes[1,1].set_xlabel("Episode")
    axes[1,1].set_ylabel("Average SNR Reduction")
    axes[1,1].legend()

    plt.tight_layout()
    st.pyplot(fig)
