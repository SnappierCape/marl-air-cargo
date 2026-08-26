# ✈️ Multi-Agent Reinforcement Learning in Schiphol Airport Landside Cargo Logistics

[![Project Status: Complete](https://img.shields.io/badge/Project%20Status-Complete-green.svg)]()
[![Engine: SimPy](https://img.shields.io/badge/Simulation-SimPy-blue.svg)]()
[![Interface: PettingZoo](https://img.shields.io/badge/Interface-PettingZoo-orange.svg)]()
[![RL Engine: BenchMARL](https://img.shields.io/badge/RL%20Engine-BenchMARL-red.svg)]()
[![Results Tracking: WandB](https://img.shields.io/badge/Results%20Tracking-WandB-yellow.svg)]()

> A high-fidelity **Multi-Agent Reinforcement Learning (MARL)** environment designed to optimize truck slot bookings and landside congestion at Amsterdam Airport Schiphol Cargo Hub. This project bridges discrete-event simulation (DES) with modern deep MARL frameworks leveraging the **Multi Agent Proximal Policy Optimization (MAPPO)** algorithm.

---

## 📦 Schiphol Cargo Hub Background

The Schiphol Cargo Hub is a primary European logistical gateway with high-density freight traffic managed by several independent Ground Handling Agents (GHAs). Landside operations currently rely on a decentralized model where GHAs manage their own warehouse and dock infrastructures. A primary operational challenge is the absence of a unified Truck Slot Booking System to synchronize inbound truck traffic with available dock capacity.

During peak periods, such as the concentrated arrival patterns on Friday afternoons, high logistics volume often exceeds immediate infrastructure capacity. Without a centralized layer to align slot availability with truck arrivals, the system can experience a temporal mismatch. This leads to uneven dock utilization and increased dwell times for transport vehicles.

This project uses a discrete-event simulation to study whether Multi-Agent Reinforcement Learning (MARL) can address this coordination gap. By modeling the hub as a multi-agent environment, the research evaluates if autonomous agents can learn to synchronize scheduling and resource allocation in a decentralized way. The objective is to determine if MARL can effectively transform independent decision-making into a cohesive logistical flow to improve the overall efficiency of the Schiphol cargo ecosystem.

---

## 🏗️ Project Architecture

This project implements a **Centralized Training, Decentralized Execution (CTDE)** paradigm. It models the complex interactions between independent Ground Handling Agents (GHAs), a central Transporter agent, and an optional Airport Orchestrator.

---

### 🧩 Core Modules
| Module | Responsibility |
| :--- | :--- |
| **`dtp_platform.py`** | The "Digital Twin" platform rules; manages slot publishing, booking, and validation. |
| **`objects.py`** | Physical entities: `Trucks`, `GHATerminal` (Docks/Queues), and `TP3Buffer` (Parking). |
| **`demand.py`** | Stochastic arrival engine; manages the lifecycle and journey of every truck. |
| **`infrastructure.py`** | The sensor layer; tracks ANPR events and timestamps for KPI calculation. |
| **`kpi_tracker.py`** | Translates raw events into rewards (WPR, Utilization, etc...). |
| **`road.py`** | Calculates intra-airport travel times between each node based on real-world distances. |
| **`service_time.py`** | Applies random noise to the dock service time. |
| **`schiphol_env.py`** | PettingZoo env wrapper to bridge with multiple MARL engines. |
| **`train.py`** | Main training script. |
| **`benchmarl_task.py`** | Bridges the PettingZoo environment and the BenchMARL task. |

---

### 🪜 Layered Structure

The project is logically organized into 3 "Layers" so that every layer acts as a base for the next one, there is no clear layer separation among modules, but the logical layers are spread across the modules:

- **Platform Layer (the foundations):** The first layer enforces the DTP (Digital Truck Slot Planning) rules through pure Python logic, handles slot publication and slot booking, and gives dock utilization info.
- **Simulation Layer (the logistics):** The second layer leverages `SimPy` to build a simulation environment that allows to iterate layer 1 through tens of thousands of episodes, in order to give to the MARL engine something to learn from.
- **MARL Layer (the brain):** The last layer is the Reinforcement Learning engine, which leverages `BenchMARL` to allow the agents to learn smart policies from the SimPy simulations.

---

## 📂 Project Structure

```text
.
├── config/             # YAML/Python global parameters
├── env/                # Core Simulation Logic
│   ├── schiphol_env.py
│   ├── objects.py
│   ├── dtp_platform.py
│   └── ...            
├── testing/            # Sanity checks and execution scripts
├── marl/               # Reinforcement Learning adapters
└── scripts/            # Training script
```

---

## ⚙️ Technology Stack

The project is implemented in `Python 3.12`. The following external libraries are used:

| Library        | Scope                                               |
|----------------|-----------------------------------------------------|
| **Numpy**      | High-performance vectorial math                     |
| **SimPy**      | Simulations                                         |
| **Gymnasium**  | Agents' action spaces                               |
| **PettingZoo** | SimPy wrapper for bridging simulations with RL      |
| **TorchRL**    | MARL Engine                                         |
| **BenchMARL**  | TorchRL wrapper for ease of use and reproducibility |
| **Hydra**      | Command line utility                                |
| **WandB**      | Results summary and comparison                      |

---

## 🏅 Reward Formulation & Incentive Engineering

To support the **Centralized Training, Decentralized Execution (CTDE)** framework, the environment uses a hybrid reward structure. Because different actors in the Schiphol landside cargo ecosystem have conflicting operational priorities (e.g., a GHA wants to optimize its own docks, while the Transporter wants to minimize total fleet transit delay), the environment implements an $\alpha$-blended mixed incentive model.

---

### 1. Private vs. Global Rewards

Every agent $i$ receives a synthesized step-wise reward $R_i$ composed of two separate vector components:

#### A. Private Rewards ($R_{\text{private}, i}$)
These track local, agent-centric operational metrics. They penalize or reward behaviors directly controlled by or affecting that specific entity.

#### B. Global Reward ($R_{\text{global}}$)
This is a shared macro-signal distributed identically to all active agents. It tracks system-wide coordination, ecosystem welfare, and absolute hub stability. In this case it is based on:
- **Dock Utilization StDev:** Standard deviation of dock utilization across all the GHAs as a measure of load balancing.
- **Wait-to-Process Ratio (WPR):** The fraction of time that the trucks spends idling.

---

### 2. The $\alpha$-Blending Mechanics

To balance competitive individual performance with holistic network cooperation, the environment computes the final scalar reward for agent $i$ at each step using a linear combination parameter, $\alpha$:

$$R_i = \alpha \cdot R_{\text{private}, i} + (1 - \alpha) \cdot R_{\text{global}}$$

Where $\alpha \in [0, 1]$ represents the **Selfishness Coefficient**. 

```
┌────────────────────────┐
│  Private Reward (R_i)  │───┐
└────────────────────────┘   │    (x α)
                             ├───────────> [ (+) Combined Agent Reward ]
┌────────────────────────┐   │  (x 1-α)
│   Global Reward (R_g)  │───┘
└────────────────────────┘
```

The system behaves under three distinct operational paradigms depending on your configuration file (`config/params.yaml`):

| Configuration Mode | Value of $\alpha$ | Behavioral Outcome | MARL Dynamic |
| :--- | :--- | :--- | :--- |
| **Fully Cooperative** | $\alpha = 0.0$ | All agents optimize strictly for total hub throughput and global traffic reduction. Agents will actively sacrifice individual efficiency if it prevents downstream system gridlock. | Pure Coordination (Social Welfare) |
| **Purely Competitive** | $\alpha = 1.0$ | Agents act as completely selfish utility-maximizers. GHAs ignore global congestion to force dock utilization spikes; the Transporter overbooks slots to minimize its individual delay metrics. | Decentralized Non-Cooperative Game |
| **Mixed-Incentive** | $0.0 < \alpha < 1.0$ | **(Default)** Agents seek local operational excellence while maintaining systemic boundaries. A GHA optimizes local docks but avoids greedily scheduling actions that degrade global hub traffic conditions. | Mixed Settings |

---

## 🚀 Current Milestone: [4/5] Environment Engineering
At the moment the project is fully operational and the training run has been done. Future work could address benchmarking and sensitivity analysis to parameters.

- [x] DTP Rules: Implementation of the slot booking constraints.

- [x] Core Simulation: Discrete event logic for cargo handling and truck movement.

- [x] PettingZoo Wrapper: Full implementation of observation_space, action_space, and action_masking.

- [x] BenchMARL Adapter: Full translation of PettingZoo objects into BenchMARL tasks.

- [x] MAPPO Run: Training MAPPO agents using the BenchMARL framework.

- [ ] Future Enhancements: Multi-scenario benchmarking (Scenario M vs. Scenario MO), sensitivity analysis, benchmarks against rule-based algorithms.

---

## 🛠️ Installation & Usage
This project is implemented on ubuntu 24.04 using `UV` as a package manager, for any implementation with on different platforms please consult official docs and make sure to check out the `pyproject.toml` file for library dependencies.

Installation steps:

  1. Clone the GitHub repository
```Bash
git clone https://github.com/SnappierCape/ds-marl-air-cargo.git
cd /your-path/ds-marl-air-cargo
git fetch origin
git pull origin main
```

  2. Setup UV Environment
```Bash
uv lock
uv sync
```

  3. Run a Sanity Check to verify that the PettingZoo wrapper is correctly communicating with the SimPy engine:
```Bash
uv run ./scripts/simulation.py --steps=2000 --orchestrator
```

  4. Run the MARL training loop:
```Bash
uv run ./scripts/train.py
```

---

## 📜 Acknowledgments

Developed for researchers and logistics engineers interested in the intersection of Operations Research and Machine Learning.

Special thanks to the Schiphol Landside Cargo community for the operational insights.