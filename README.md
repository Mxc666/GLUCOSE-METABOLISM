## Data Sources

* **TCGA / GTEx** RNA-seq and clinical metadata (GDC portal, GTEx portal).
* **KEGG** KGML pathway files (human, `hsa*.xml`) for stoichiometry and GPR.
* **BRENDA**/**UniProt**/**CatPred** for curated or predicted *k*₍cat₎ priors.
* Gene sets for downstream glucose pathways, lactate export, and lactylation (CSV files in `biological-process-genes` and `Lactation-Data`).

---

## Software Requirements

The code was developed and tested on **Python 3.10** in a Conda environment. Key packages and versions (pinned to your system):

* `pandas==2.2.2`, `numpy==1.26.4`, `scipy==1.15.3`, `statsmodels==0.14.5`
* `scikit-learn==1.7.1`, `matplotlib==3.10.3`, `seaborn==0.13.2`
* `bioservices==1.11+` (installed via `pip`, version managed by PyPI)
* `mygene==3.2.2`, `gseapy==1.1.9`
* Optimization: `osqp==1.0.4`, `pulp==3.2.1`, (optional) `cvxpy==1.4.4`
* Optional modeling/omics: `cobra==0.29.1`, `igraph==0.11.9`
