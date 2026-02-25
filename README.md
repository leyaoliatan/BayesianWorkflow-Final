# BayesianWorkflow-Final

1. Add your real-world data paths first in:
`results/application/input_data_paths.txt`

Use one line per dataset:
`category:/path/to/your_data.csv`

2. Install dependencies:
`pip install -r requirements.txt`

3. Run notebooks in this order:
- `notebooks/01_simulation_experiments.ipynb`
- `notebooks/01_simulation_experiments_cd.ipynb`
- `notebooks/02_real_data_experiments.ipynb`
- `notebooks/03_paper_tabs.ipynb`
- `notebooks/04_type2map_vs_mcmc_A_seed0.ipynb`

4. Check outputs under:
- `results/simulations/`
- `results/application/`
- `results/summary_simulations/`
- `results/summary_applications/`
