# ULDM-BinaryPulsars-DNN

Using deep neural networks, we probe how well binary pulsars can detect or constrain ultra-light dark matter (ULDM).

We have developed a code that simulates pulsar timing residuals. The residuals contain both stochastic contributions (Gaussian noise — either white or white plus red) and deterministic signals. The deterministic component may contain nuisance signals, ULDM signals, or a combination of both, depending on the signal complexity and the methods applied.

To determine the sensitivity, we employ three machine learning techniques:

1. **Autoencoder** — used as an anomaly detector.
2. **Binary classifier** — to distinguish between noise and signal.
3. **Multiclass classifier** — to demonstrate that four types of ULDM-induced signals can be distinguished from each other and from noise.

This repository provides code, data generation tools, and models to support these analyses.

## Authors

- Pavel Kůs — [pavel.kus@fzu.cz](mailto:pavel.kus@fzu.cz)  
- Diana López Nacir — [dnacir@df.uba.ar](mailto:dnacir@df.uba.ar)  
- Federico R. Urban — [federico.urban@fzu.cz](mailto:federico.urban@fzu.cz)
