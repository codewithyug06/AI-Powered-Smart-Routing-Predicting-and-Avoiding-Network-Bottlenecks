# AI-Powered-Smart-Routing-Predicting-and-Avoiding-Network-Bottlenecks

---

## 📘 Overview

Traditional routing algorithms such as **Dijkstra’s** or **A\*** are optimal for static graphs but fail under **dynamic congestion** and **spatio-temporal bottlenecks**.  
This project introduces an **AI-powered routing system** that learns to avoid congestion dynamically through intelligent **bottleneck detection, prediction, and adaptive routing**.

The framework leverages:
- 🧩 **GCN/GAT** for topological bottleneck detection (spatial domain)
- ⏱ **LSTM** for time-series bottleneck forecasting (temporal domain)
- 🎯 **Curriculum-based D-DQN** for adaptive routing decisions
- ⚙️ **Integration with classical SSSP (Dijkstra)** to maintain stability and interpretability

---

# 📊 Dataset Information

**Dataset Name:** [5G Traffic Datasets](https://www.kaggle.com/datasets/kimdaegyeom/5g-traffic-datasets)  
**Author:** Daegyeom Kim  
**Platform:** Kaggle  
**Year:** 2023  

### Description
A large-scale real-world 5G traffic dataset captured directly from mobile terminals using **PCAPdroid** and **G-NetTrack Pro** without specialized hardware.  
It includes **328 hours** of timestamped 5G packet data (CSV format) recorded from major **video streaming**, **live streaming**, **video conferencing**, **metaverse**, and **gaming** applications — ideal for **network analysis**, **GNN modeling**, and **deep learning-based routing**.

### Key Highlights
- Captured on **Samsung Galaxy A90 5G** (Qualcomm Snapdragon X50 modem)  
- Traffic types: Netflix, YouTube Live, Zoom, Roblox, MS Teams, etc.  
- Each record includes packet headers, source/destination IPs, timestamps, and payload info  
- Suitable for **GANs**, **LSTMs**, and **spatio-temporal ML models**

### Structure
- `packets.csv` — Timestamped packet-level 5G traffic data  
- `applications.csv` — Application-wise traffic statistics  
- `meta_info.json` — Device and capture configuration details  

📥 **To Use:**
1. Download from [Kaggle → 5G Traffic Datasets](https://www.kaggle.com/datasets/kimdaegyeom/5g-traffic-datasets)  
2. Extract to the `/data/5g_traffic_datasets/` directory of this repository  
3. Update dataset path in your configuration file if required
---

## 🧩 Key Features

| Feature | Description |
|----------|--------------|
| 🧠 **Hybrid AI Routing** | Combines classical SSSP with AI for real-time decision making |
| 🕸 **Graph Neural Networks** | GCN & GAT for spatial bottleneck localization |
| ⏱ **Temporal Prediction** | LSTM-based spatio-temporal bottleneck forecasting |
| 🎮 **Reinforcement Learning** | D-DQN with dueling architecture & curriculum learning |
| 📈 **Massive Performance Gain** | 7×–53× improvements in success, throughput, and latency |
| ⚡ **5G Network Simulation** | Tested on realistic 5G traffic topologies |

---

## 🏗️ System Architecture


             ┌──────────────────────────────┐
             │     Network Topology G(V,E)  │
             └──────────────┬───────────────┘
                            │
            ┌───────────────┴───────────────┐
            │         Bottleneck Detection  │
            │           (Spatial)           │
            │                               │
            │    ┌──────────────┐           │
            │    │   GCN / GAT  │ → Spatial │
            │    └──────────────┘           │
            └──────────────┬────────────────┘
                           │
            ┌──────────────┴───────────────┐
            │        Prediction (Temporal) │
            │                              │
            │    ┌──────────────┐          │
            │    │    LSTM      │ → Temporal│
            │    └──────────────┘          │
            └──────────────┬────────────────┘
                           │
            ┌──────────────┴───────────────┐
            │      Routing Decision        │
            │                              │
            │  ┌────────────────────────┐  │
            │  │  Curriculum D-DQN      │ → Adaptive │
            │  └────────────────────────┘  │
            └──────────────────────────────┘


---

## ⚙️ Implementation Details

| Component | Details |
|------------|----------|
| **Dataset** | 5G traffic dataset (10–200 nodes/snapshot) |
| **Graph Features** | 12 node features (degree, load, latency, etc.) |
| **Loss Function** | Focal loss (for imbalance), Cross-Entropy |
| **Optimizer** | AdamW with cosine annealing |
| **Learning Rate** | `1e-4 → 1e-6 (decay 0.9995)` |
| **Batch Size** | 128 |
| **Episodes** | 5,000 with 200 warmup |
| **Frameworks** | PyTorch, PyTorch Geometric, NetworkX |

---

## 📊 Experimental Results

| Algorithm | Success (%) | Latency (ms) | Throughput (paths/s) | Bottleneck Hits | Risk-Free (%) |
|------------|-------------|---------------|-----------------------|------------------|----------------|
| **GCN-DDQN** | 🥇 **99.5** | 0.04 | 3,908 | 0.60 | 45.7 |
| **GAT-DDQN** | 98.0 | 0.27 | 4,422 | **0.53** | **50.0** |
| **Dijkstra** | 98.5 | **0.07** | 8,595 | 0.67 | 41.1 |
| **A*** | 98.0 | 0.05 | 10,697 | 0.71 | 35.7 |
| **DRL-GNN [2]** | 12.5 | 21.15 | 73.2 | 0.80 | 36.0 |
| **GAT-RL [3]** | 18.5 | 5.95 | 106.1 | 0.73 | 35.1 |

**Highlights:**
- 🧩 99% spatial detection (GCN)
- 🕒 100% temporal prediction (GAT-LSTM)
- ⚡ 60× throughput vs DRL-GNN
- 🔄 22–529× latency reduction

---

## 🧩 Methodological Workflow

1. **Graph Representation:**  
   - Network → Directed Graph (Nodes = Routers, Edges = Links)
2. **Spatial Detection (GCN/GAT):**  
   - Detect topological bottlenecks using learned embeddings
3. **Temporal Forecasting (LSTM):**  
   - Predict future congestion probabilities (1–2 steps ahead)
4. **Adaptive Routing (D-DQN):**  
   - Optimize paths using curriculum-guided double Q-learning
5. **Risk-Aware Path Update:**  
   - Integrate spatial and temporal risk penalties for decision-making

---

## 🧪 Evaluation Metrics

| Category | Metrics |
|-----------|----------|
| **Routing** | Success rate, path cost, latency, throughput |
| **Risk** | Bottleneck hits, spatial/temporal risk scores |
| **Computation** | FLOPs, runtime, decision latency |

---

## 🧮 Reward and Learning Strategy

**Reward Function Components**
- ✅ Reaching destination  
- 🚫 Bottleneck avoidance  
- ⚡ Path efficiency  
- 🔄 Step penalty  
- 🎯 Risk minimization (spatial & temporal)  

---

