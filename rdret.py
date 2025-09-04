# -*- coding: utf-8 -*-
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

# ==========================================
# Ortam Tanımı
# ==========================================
class RealisticJammingEnv(AECEnv):
    def __init__(self, n_jammers=2, n_radars=1, max_power=10,
                 n_freq=5, pw_min=1, pw_max=5, pri_min=2500, pri_max=2530):
        super().__init__()
        self.n_jammers = n_jammers
        self.n_radars = n_radars
        self.agents = ["jammer_" + str(i) for i in range(n_jammers)] + ["radar_" + str(i) for i in range(n_radars)]
        self.possible_agents = self.agents[:]
        self.max_power = max_power
        self.n_freq = n_freq
        self.pw_min = pw_min
        self.pw_max = pw_max
        self.pri_min = pri_min
        self.pri_max = pri_max

        # Action space: power * freq kombinasyonu
        self.action_spaces = {agent: spaces.Discrete(max_power * n_freq) for agent in self.agents}
        # Observation: [power, freq, SNR, PW, PRI]
        self.observation_spaces = {agent: spaces.Box(low=0, high=max_power, shape=(5,), dtype=np.float32)
                                   for agent in self.agents}

    def reset(self, seed=None, options=None):
        self.state = {agent: np.array([
            0.0,
            np.random.randint(self.n_freq),
            np.random.rand()*5,
            np.random.uniform(self.pw_min, self.pw_max),
            np.random.uniform(self.pri_min, self.pri_max)
        ], dtype=np.float32) for agent in self.agents}
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
            self.state[agent][0] = power
            self.state[agent][1] = freq

            reward = 0
            if "jammer" in agent:
                for r in [a for a in self.agents if "radar" in a]:
                    if freq == int(self.state[r][1]):
                        self.state[r][2] = max(0, self.state[r][2] - 0.1*power)
                reward = np.sum([5 - self.state[r][2] for r in self.agents if "radar" in r])
            else:
                reward = self.state[agent][2]

            self._cumulative_rewards[agent] += reward
            rewards[agent] = reward
        return self.state, rewards

# ==========================================
# DQN Agent
# ==========================================
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

# ==========================================
# Streamlit Sayfaları
# ==========================================
page = st.sidebar.selectbox("Choose a page", ["Simulation", "Explanation"])

if page == "Explanation":
    st.title("Program Açıklaması")
    st.markdown("""
    **Multi-Agent DQN ET Jamming Simülasyonu**

    Bu program, elektromanyetik çevreye (Electronic Warfare) yönelik **Multi-Agent DQN (Deep Q-Learning)** tabanlı bir simülasyonu görselleştirir.

    ### Amaç:
    - Bir veya birden fazla **jammer** radar sinyallerini bozmak için strateji öğrenir.
    - Radar ajanları ise sinyallerini koruyacak şekilde frekans ve diğer parametreleri seçer.

    ### Başarı Kriterleri:
    - **Radar için:** Yüksek Detection Score (SNR üzerinden normalize edilmiş performans).
    - **Jammer için:** Yüksek Ortalama Jamming Impact (radarların SNR düşüşü).

    ### Simülasyon Mantığı:
    - Her ajan belirli bir state gözlemi alır: `[Power, Frequency, SNR, PW, PRI]`
    - Jammer ajanlar radarın SNR’ını düşürmeye çalışır.
    - Radar ajanlar sinyalini korumaya çalışır.
    - DQN algoritması ajanlara kendi policy’lerini öğrenme imkanı verir.
    - Sonuçlar tablolarda ve heatmap’lerde görselleştirilir.
    """)

elif page == "Simulation":
    st.title("Multi-Agent DQN ET Jamming Simulation")

    # Sidebar parametreleri
    n_episodes = st.sidebar.slider("Number of Episodes", 50, 500, 200, 25)
    n_jammers = st.sidebar.slider("Number of Jammers", 1, 5, 2)
    n_radars = st.sidebar.slider("Number of Radars", 1, 3, 1)
    max_power = st.sidebar.slider("Max Jammer Power", 1, 20, 10)
    n_freq = st.sidebar.slider("Number of Frequencies", 2, 10, 5)
    pw_min = st.sidebar.slider("PW Min (µs)", 1, 10, 2)
    pw_max = st.sidebar.slider("PW Max (µs)", 1, 10, 5)
    pri_min = st.sidebar.slider("PRI Min (µs)", 1000, 5000, 2500)
    pri_max = st.sidebar.slider("PRI Max (µs)", 1000, 5000, 2530)

    start_button = st.sidebar.button("Start Simulation")

    if start_button:
        env = RealisticJammingEnv(n_jammers, n_radars, max_power, n_freq, pw_min, pw_max, pri_min, pri_max)
        agents = {agent: DQNAgent(obs_size=5, n_actions=env.action_spaces[agent].n) for agent in env.agents}

        snr_baseline = 5.0
        snr_history = {r: [] for r in env.agents if "radar" in r}
        detection_scores = {r: [] for r in snr_history}
        jammer_freq_history = {j: [] for j in env.agents if "jammer" in j}
        radar_freq_history = {r: [] for r in snr_history}
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
                radar_freq_history[r].append(int(next_state[r][1]))
            for j in jammer_freq_history:
                jammer_freq_history[j].append(int(next_state[j][1]))

            jamming_impact = np.mean([snr_baseline - next_state[r][2] for r in snr_history])
            jamming_impact_scores.append(jamming_impact)

            progress_bar.progress((ep + 1)/n_episodes)
            status_text.text(f"Episode {ep+1}/{n_episodes}: Avg Jamming Impact = {jamming_impact:.2f}")
            time.sleep(0.01)

        st.success("Simulation Complete!")

        # Başarı kriterleri
        radar_success = {r: {"Avg SNR": np.mean([state[r][2] for state in state_history]),
                             "Avg Detection Score": np.mean(detection_scores[r])} for r in snr_history}

        jammer_success = {j: np.mean([np.mean([snr_baseline - state[r][2] for r in snr_history])
                                      for state in state_history]) for j in jammer_freq_history}

        best_radar = max(radar_success, key=lambda x: radar_success[x]["Avg Detection Score"])
        best_jammer = max(jammer_success, key=lambda x: jammer_success[x])

        st.subheader("Başarı Kriterleri")
        st.write("**Radar Ajanları Başarı Durumu**")
        st.table(radar_success)
        st.write("**Jammer Ajanları Başarı Durumu (ortalama Jamming Impact)**")
        st.table({j: {"Avg Jamming Impact": jammer_success[j]} for j in jammer_success})
        st.success(f"En Başarılı Radar: {best_radar} (yüksek Detection Score)")
        st.success(f"En Başarılı Jammer: {best_jammer} (yüksek Jamming Impact)")

        # Dashboard grafikleri


        fig, axes = plt.subplots(3,2, figsize=(16,12))  # 3 satır 2 sütun
        
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
        axes[0,1].set_title("Radar Detection Score")
        axes[0,1].legend()
        
        # Jammer Frequency Heatmap
        jammer_matrix = np.array([jammer_freq_history[j] for j in jammers])
        sns.heatmap(jammer_matrix, annot=False, fmt="d", cmap="YlOrRd", cbar=True, ax=axes[1,0])
        axes[1,0].set_xlabel("Episode")
        axes[1,0].set_ylabel("Jammer Agents")
        axes[1,0].set_title("Jammer Frequency Choices")
        
        # Radar Frequency Heatmap
        radar_matrix = np.array([radar_freq_history[r] for r in radars])
        sns.heatmap(radar_matrix, annot=False, fmt="d", cmap="Blues", cbar=True, ax=axes[1,1])
        axes[1,1].set_xlabel("Episode")
        axes[1,1].set_ylabel("Radar Agents")
        axes[1,1].set_title("Radar Frequency Choices")
        
        # -------------------------------
        # Avg Jamming Impact Grafiği
        # -------------------------------
        axes[2,0].plot(jamming_impact_scores, color='red', label='Avg Jamming Impact')
        axes[2,0].set_xlabel("Episode")
        axes[2,0].set_ylabel("Average SNR Reduction")
        axes[2,0].set_title("Avg Jamming Impact Over Episodes")
        axes[2,0].legend()
        
        # Boş bırakılacak yer (opsiyonel)
        axes[2,1].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)

