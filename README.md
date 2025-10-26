# Continuous‑Phase Hamiltonian Learning

Fully modular **Julia** codebase for **continuous‑phase Hamiltonian learning (CPHL)** in the context of **inverse quantum simulation**. A corresponding preprint will be released on **arXiv soon**.

The key idea of the algorithm is to **extend a phase of matter of interest via a continuous modification of the original Hamiltonian**. We illustrate the algorithm on the **1D Cluster Ising model (CIM)**, where we extend its **topological phase**.

To reproduce the main results:

* **Run `ContPhaseLearn_DMRG.ipynb`** — executes the algorithm where the quantum circuit is replaced with **DMRG** and **matrix product states**.
* **Run `Plotter_CPHL.ipynb`** — performs refined visualization and plotting.

These results are summarized in **Fig. 3** of the main text and detailed in the **Supplementary Information (CPHL section)** of the forthcoming preprint.
