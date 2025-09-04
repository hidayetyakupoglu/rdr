# -*- coding: utf-8 -*-
# ==========================================
# Multi-Agent DQN ET Jamming - Multi-Page Streamlit App
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
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import AgentSelector
from gymnasium import spaces

# -------------------------------
# Ortam Tanımı
# -------------------------------
class RealisticJammingEnv(AECEnv):
    def __init__(self, n_jammers=2, n_radars=1, max_power=10, n_freq=5,
                 pw_range=(2.0,3.0), pri_range=(2500.0,2532.0)):
        super().__init__()
        self.n_jammers = n_jammers
        self.n_radars = n_radars
        self.agents = ["jammer_" + str(i) for i in range(n_jammers)] + ["radar_" + str(i) for i in range(n_radars)]
        self.possible_agents = self.agents[:]
        self.max_power = max_power
        self.n_freq = n_freq
        self.pw_range = pw_range
        self.pri_range = pri_range

        self.action_spaces = {agent: spaces.Discrete(max_power * n_freq) for agent in self.agents}
        self.observation_spaces = {agent: spaces.Box(low=0, high=max_power, shape=(5,), dtype=np.float32)
                                   for agent in self.agents}

    def reset(self, seed=None, options=None):
        self.state = {
            agent: np.array([
                0.0,
                np.random.randint(self.n_freq),
                np.random.rand()*5,
                np.random.uniform(self.pw_range[0], self.pw_range[1]),
                np.random.uniform(self.pri_range[0], self.pri_range[1])
            ], dtype=np.float32)
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
            self.state[agent][:2] = [power, freq]

            reward = 0
            if "jammer" in agent:
                for r in [a for a in self.agents if "radar" in a]:
                    freq_match = freq == int(self.state[r][1])
                    pw_factor = np.clip(self.state[r][3] / self.pw_range[1], 0.5, 1.0)
                    pri_factor = np.clip(self.pri_range[0] / self.state[r][4], 0.5, 1.0)
                    reduction = 0.1 * power * freq_match * pw_factor * pri_factor
                    self.state[r][2] = max(0, self.state[r][2] - reduction)
                reward = np.sum([5 - self.state[r][2] for r in self.agents if "radar" in r])
            else:
                reward = self.state[agent][2]

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
# Sayfa Seçimi
# -------------------------------
page = st.sidebar.selectbox("Choose Page", ["Simulation", "Explanation"])

if page == "Explanation":
    st.title("Program Açıklaması")
    st.markdown("""
    ## Multi-Agent DQN ET Jamming Simulation

    Bu uygulama, **Elektronik Taarruz (Electronic Attack / EA) jamming** senaryosunu simüle etmek için geliştirilmiştir.
    Programda iki tip ajan vardır:

    1. **Jammer Ajanları**: Radarların sinyalini bozmaya çalışır.
    2. **Radar Ajanları**: Sinyalini korumaya çalışır.

    ### Kullanılan Yöntemler:
    - **DQN (Deep Q-Network)** algoritması ile ajanlar kendi eylemlerini öğrenir.
    - **PettingZoo** ortamı multi-agent simülasyon için kullanılır.
    - Her ajan kendi **observation vector**'üne sahiptir:
        - `[Power, Frequency, SNR, Pulse Width (PW), Pulse Repetition Interval (PRI)]`

    ### Kullanıcı Parametreleri:
    - **Jammer sayısı** ve **Radar sayısı**
    - **Jammer gücü** (Power)
    - **Frekans sayısı**
    - **PW min/max** ve **PRI min/max** değerleri

    ### Çıktılar:
    - Radar SNR grafikleri
    - Jammer frekans seçimleri heatmap’i
    - Detection score ve jamming impact skorları
    - Hangi ajanların daha başarılı olduğu tablo
    """)

elif page == "Simulation":
    st.title("Multi-Agent DQN ET Jamming Simulation")
    # -------------------------------
    # Sidebar Parametreleri
    # -------------------------------
    n_episodes = st.sidebar.slider("Number of Episodes", 50, 500, 200, 25)
    n_jammers = st.sidebar.slider("Number of Jammers", 1, 5, 2)
    n_radars = st.sidebar.slider("Number of Radars", 1, 3, 1)
    max_power = st.sidebar.slider("Max Jammer Power", 1, 20, 10)
    n_freq = st.sidebar.slider("Number of Frequencies", 2, 10, 5)

    pw_min = st.sidebar.number_input("PW Min (µs)", 0.1, 10.0, 2.0)
    pw_max = st.sidebar.number_input("PW Max (µs)", pw_min, 10.0, 3.0)
    pri_min = st.sidebar.number_input("PRI Min (µs)", 1000.0, 5000.0, 2500.0)
    pri_max = st.sidebar.number_input("PRI Max (µs)", pri_min, 5000.0, 2532.0)

    st.sidebar.write("Click 'Start Simulation' to run.")
    start_button = st.sidebar.button("Start Simulation")

    # -------------------------------
    # Simülasyonu Başlat
    # -------------------------------
    if start_button:
        env = RealisticJammingEnv(n_jammers=n_jammers, n_radars=n_radars, max_power=max_power,
                                  n_freq=n_freq, pw_range=(pw_min,pw_max), pri_range=(pri_min,pri_max))
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
        # Dashboard
        # -------------------------------
        fig, axes = plt.subplots(2,2, figsize=(14,10))
        radars = list(snr_history.keys())
        jammers = list(jammer_freq_history.keys())
        state = state_history[-1]

        # Radar SNR
        axes[0,0].bar(radars, [state[r][2] for r in radars], color='skyblue')
        axes[0,0].set_ylim(0,5)
        axes[0,0].set_title("Radar SNR at Final Episode")

        # Detection Score
        for r in radars:
            axes[0,1].plot(detection_scores[r], label=r)
        axes[0,1].set_xlabel("Episode")
        axes[0,1].set_ylabel("Detection Score")
        axes[0,1].legend()
        axes[0,1].set_title("Radar Detection Score")

        # Jammer Frequency Heatmap
        jammer_matrix = np.array([jammer_freq_history[j] for j in jammers])
        sns.heatmap(jammer_matrix, annot=False, fmt="d", cmap="YlOrRd", cbar=True, ax=axes[1,0])
        axes[1,0].set_xlabel("Episode")
        axes[1,0].set_ylabel("Jammer Agents")
        axes[1,0].set_title("Jammer Frequency Choices")

        # Jamming Impact Score
        axes[1,1].plot(jamming_impact_scores, color='red', label='Jamming Impact')
        axes[1,1].set_xlabel("Episode")
        axes[1,1].set_ylabel("Avg SNR Reduction")
        axes[1,1].set_title("Jammer Impact Score")
        axes[1,1].legend()

        plt.tight_layout()
        st.pyplot(fig)
