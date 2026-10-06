### Entropy-Based Network Intrusion Detection using Markov Chains

hello, 

this is the project I worked on as part of the "Information Theory (ECE_TEL851)" course from the university of Peloponnese and its a Network Intrusion Detection System (NIDS).

---

**Overview:**

This project proposes a lightweight model which is "trained" on normal (or "healthy") network traffic and therefore easily detects anomalies on a **specific** network and its type of traffic. The training is done by studying the metadata (volume of data, connection duration, etc.). 

**More Detailed Explaination:**

The system analyzes network traffic by converting packet sequences into stochastic processes. It monitors the state transitions of various network features (such as Source/Destination IP, Ports, and TCP Flags) and flags sudden shifts as potential intrusions.

The fun part of the matter is that instead of using complex algorithms actually trained on the data, we simply create state transition probability matrices (based on 1st-order Markov Chains) for normal traffic. Then we base our judgement for what's coming on deviations from metrics like the systems' Shannon Entropy (baseline). 

Of course the system as is, comes with some deal-breaker issues in real life/time, but this is just a simple mathematical approach to a new area for me and an interesting concept, straying away from common firewalls.

---

#### Key Features (Buzzwords):
- **Markov Chain Modeling**: We utilize first-order Markov Chains to profile the normal behavior of a network.
- **Entropy-Based Detection**: Measures deviations from the expected information uncertainty using Shannon and Conditional Entropy.
- **Explainability**: Mathematically transparent, allowing anyone to trace exactly *why* an alert was triggered.
- **Huffman Coding Integration**: We use compression algorithms as a secondary indicator for anomalous payload patterns *(still working on it)*.


---

#### Repository Structure:

- `code pipeline (notebooks)/`: Contains the core Python/Jupyter scripts used for data preprocessing, model training, and testing.\ 

	- `BETA (to be continued...)`/: Some work to continue building this project. Its got the Huffman Evaluation script, a semi-functional web-app built with Streamlit and other things.

- `Entropy-Based Network Intrusion Detection using Markov Chains (ENG).pdf`: The complete academic report (English version).
- `Entropy-Based Network Intrusion Detection using Markov Chains (GRE).pdf`: The complete academic report (Greek version).
