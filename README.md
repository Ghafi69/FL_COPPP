# FL-COP++: Layered Pre-Aggregation Defense for Federated Learning

FL-COP++ is a MATLAB framework for studying data-state consistency, malicious model updates, robust aggregation, and communication efficiency in federated learning (FL). It combines an authenticated enrollment protocol, nonce-bound Merkle–HMAC challenges, behavioral update screening, client trust management, robust aggregation, and sparse/quantized model updates.

The implementation supports experiments with **CIFAR-10** and **FEMNIST**, including clean training, post-enrollment label modification, poisoned-at-enrollment backdoors, model replacement, and sign-flipping attacks.

## 1\. Method overview

FL-COP++ uses two complementary defense stages:

1. **Before local training:** each client registers a canonical representation of its local state. Sampled Merkle–HMAC challenges test whether the current serialized state remains consistent with the enrolled commitment.
2. **Before aggregation:** direction, update magnitude, and prediction-drift signals are combined with evolving client trust and robust aggregation to reduce the influence of anomalous updates.

The framework also supports top-*k* sparsification and low-bit quantization of client updates. These encoding mechanisms reduce update traffic; the integrity protocol adds its own authentication and verification overhead. Communication measurements account for both effects separately.

**Security scope.** Successful enrolled-state verification establishes consistency with the registered representation, not semantic validity of the data, correct training execution, or proof that the registered data produced the submitted update. Poisoned-at-enrollment data, retained-copy evasion, and update-only manipulation are outside the direct cryptographic guarantee.

## 2\. Datasets

Dataset files are **not distributed** with the source code and must be obtained from their original providers.

|Dataset|Source|Input format|Experimental use|
|-|-|-|-|
|**CIFAR-10**|[University of Toronto](https://www.cs.toronto.edu/~kriz/cifar.html) · [MATLAB archive](https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz)|Official MATLAB batch files (`.mat`)|Ten-class classification with simulated non-IID client partitions|
|**FEMNIST**|[LEAF benchmark](https://github.com/TalwalkarLab/leaf) · [FEMNIST preparation](https://github.com/TalwalkarLab/leaf/tree/master/data/femnist)|LEAF writer-partitioned training and testing JSON shards|Sixty-two-class classification with naturally heterogeneous writers|

### Expected dataset layout

```text
FL-COP-PlusPlus/
├── data/
│   ├── cifar10/
│   │   └── cifar-10-batches-mat/
│   │       ├── data\_batch\_1.mat
│   │       ├── data\_batch\_2.mat
│   │       ├── data\_batch\_3.mat
│   │       ├── data\_batch\_4.mat
│   │       ├── data\_batch\_5.mat
│   │       ├── test\_batch.mat
│   │       └── batches.meta.mat
│   └── femnist/
│       ├── train/
│       │   └── \*.json
│       └── test/
│           └── \*.json
├── flcoppp\_config.m
├── flcoppp\_cifar10.m
├── flcoppp\_load\_femnist.m
├── flcoppp\_protocol.m
└── README.md
```

For FEMNIST, the loader also recognizes LEAF-style `data/train` and `data/test` folders. JSON files must preserve the `users`, `user\_data`, `x`, and `y` fields and writer identifiers. Unpartitioned EMNIST files are not accepted directly by the LEAF loader.

### Dataset preparation and reference data

**CIFAR-10.** The official training data are used to construct the client population. For each training seed, a deterministic, class-stratified holdout of **1,000 training examples** is reserved as the server-held reference set. The remaining records are assigned to **50 clients** using a Dirichlet non-IID partition with concentration parameter **0.5**. The official test split is used only for evaluation.

**FEMNIST.** The program identifies writer IDs appearing in both LEAF training and testing shards, selects the **50 eligible writers with the largest training sets**, and uses writer ID to resolve ties. Under the full experiment configuration, each selected writer must have at least **40 training records**. A deterministic, writer-balanced holdout of **1,000 training examples** is reserved for the server; the remaining records retain their writer-based client assignment. The selected writer IDs and reference indices are exported with the experiment logs.

**Reference-set assumption.** The server-held reference examples and labels are assumed to be **trusted and clean**. This is an explicit experimental trust assumption, not a property established by the enrollment protocol. Reference records are disjoint from client-training records and evaluation records. Behavioral screening uses up to **256 reference examples** per evaluation.

For reproducibility, the dataset source revision, LEAF preprocessing settings, client selection, and random seeds should be retained with the generated results.

## 3\. Requirements

* **MATLAB R2022b or newer**.
* **Deep Learning Toolbox** for model training and evaluation.
* A MATLAB-compatible **Java/JVM** environment for elliptic-curve key operations, SHA-256, HMAC-SHA256, and AES-GCM.
* **Parallel Computing Toolbox** is optional; a compatible GPU is recommended for large experiment matrices.
* Sufficient local storage for checkpoints, tables, logs, and figures.

## 4\. Installation and dataset paths

Clone the repository and open its root directory in MATLAB:

```bash
git clone https://github.com/Ghafi69/FL-COP-PlusPlus.git
```

```matlab
cd('path/to/FL-COP-PlusPlus')
```

By default, `flcoppp\_config.m` looks for CIFAR-10 in `data/cifar10` and FEMNIST in `data/femnist` relative to the MATLAB working directory. Alternative dataset locations can be specified with environment variables:

```matlab
setenv('FLCOPPP\_CIFAR\_ROOT', 'D:\\datasets\\cifar10');
setenv('FLCOPPP\_FEMNIST\_ROOT', 'D:\\datasets\\femnist');
```

Paths may also be set directly through `cfg.paths.cifarRoot` and `cfg.paths.femnistRoot` in `flcoppp\_config.m`. Experiment outputs are saved to the folder configured by `cfg.paths.outputRoot`, separately from the source datasets.

## 5\. Running experiments

### Preliminary checks

```matlab
RUN\_PROTOCOL\_TESTS     % Enrollment, authentication, and verification checks
CHECK\_CIFAR\_PATH       % Verify official CIFAR-10 MATLAB batch files
CHECK\_FEMNIST\_PATH     % Verify LEAF FEMNIST train/test JSON shards
```

### Short integration tests

```matlab
RUN\_SMOKE              % CIFAR-10 integration test
RUN\_FEMNIST\_SMOKE      % FEMNIST integration test
```

These short runs are intended to check the data and software pipeline, not to estimate full-experiment performance.

### Full experiments

```matlab
RUN\_FULL\_PAPER         % CIFAR-10 experiments
RUN\_FEMNIST\_PAPER      % FEMNIST experiments
```

`RUN\_ME` provides an alternative reduced experiment matrix through the `paper\_core` profile. Component and sensitivity experiments can be launched with:

```matlab
RUN\_PAPER\_WITH\_ABLATIONS
```

Full experiments can require significant computation and disk space. Results should be reported only for completed configurations, with their corresponding saved seeds and parameters.

## 6\. Experimental configuration

The default full experiment configuration is defined in `flcoppp\_config.m`.

|Parameter|Setting|
|-|-|
|Total clients|50|
|Participating clients per round|10|
|Communication rounds|60|
|Independent random seeds|17, 43, 91|
|Local epochs|1|
|Nominal malicious-client fraction|20%|
|Poisoned-data fraction, where applicable|30%|
|Commitment block size|4,096 bytes|
|Primary challenge policy|Fixed budget of **8** blocks, uniformly sampled without replacement|
|Adaptive primary probe allocation|Disabled|
|Secondary verification|Additional diagnostic challenge, with configured budget multiplier of 4|
|Behavioral screening|Cosine similarity, norm, and reference-set prediction drift; minimum 2 abnormal signals|
|Behavioral thresholding|Median/MAD-based, MAD multiplier 3.0|
|Client trust|Initial Beta parameters (1, 1); EMA 0.90; recovery 0.03|
|Robust aggregation|Symmetric trimming fraction 0.20; robust blending coefficient 0.35|
|Update encoding|Largest 30% of coefficients; 8-bit quantized values and 32-bit indices|

The actual number of distinct primary challenges is capped by the number of enrolled blocks when a client has fewer than eight blocks. Primary and diagnostic challenge counts are recorded independently.

The full configuration evaluates **FedAvg, Trimmed Mean, RFA, Krum, FLTrust, PreGateOnly, and FL-COP++** across clean training and four attack scenarios. Other methods and sensitivity settings are available as separate configurations.

## 7\. Outputs and reproducibility

Each run creates a timestamped output directory containing structured results:

```text
<configured\_output\_directory>/
└── run\_<timestamp>\_<profile>\_<dataset>/
    ├── checkpoints/
    ├── figures/
    ├── logs/
    ├── tables/
    ├── experiment\_config.mat
    ├── experiment\_config.json
    ├── system\_and\_dataset\_ledger.json
    ├── protocol\_self\_test.json
    ├── experiment\_summary.txt
    └── FLCOPPP\_complete\_results.mat
```

Depending on the experiment, tables and logs contain round-level performance, client decisions, trust trajectories, detection outcomes, per-seed summaries, communication components, run parameters, runtime measurements, and integrity benchmarks.

Results across independent runs should be summarized using **mean ± sample standard deviation**, calculated from the actual per-seed observations. Detection denominators are determined from recorded eligible client-round opportunities rather than assumed counts.

Total communication comprises global-model downlink, encoded update uplink, integrity challenge and proof traffic, and enrollment traffic. Uplink compression savings are distinguished from the additional traffic incurred by cryptographic verification. Full-state authenticated hashing, sampled HMAC, and Merkle–HMAC comparisons should be interpreted using their observed detection coverage, runtime, traffic, and server-storage requirements.

## 8\. Data and code availability

The source code is maintained in this repository. CIFAR-10 and FEMNIST are obtained from their original public dataset providers and are subject to those providers' distribution and usage conditions. A reproducible experiment should retain the source-code version, dataset preparation details, saved configuration, seed-level results, and execution logs.

