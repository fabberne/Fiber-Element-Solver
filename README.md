# Fiber Element FEM Framework

This repository contains the implementation of a flexibility-based fiber element solver developed as part of my Master's Thesis:

> **Towards Implementing and Dimensioning of Fiber Elements in Educational Software**  
> Fabio Berner, ETH Zurich, FS 2025  
> Supervisors: Prof. Dr. Eleni Chatzi, Dr. Adrian Egger, Dr. Konstantinos Vlachas

The project provides a transparent and modular Python-based framework for nonlinear finite element analysis of structures using fiber beam-column elements. It includes tools for cross-sectional and structural analysis and is intended for educational and research purposes.

## Documentation

The theoretical background, solver implementation, and usage examples (including lamina analysis and structural simulations) are thoroughly documented in the Master Thesis:

📄 [`Master_Thesis_fabberne.pdf`](./Master_Thesis_fabberne.pdf)

Additional example scripts illustrating practical applications are also included:

- `01_Example_Cross_Sectional_Porpoerties.ipynb`
- `02_Example_Lamina_analysis.ipynb`
- `03_Example_Frame_Analysis_Linear.ipynb`
- `04_Example_Frame_Analysis_Nonlinear.ipynb`
- `05_Example_Cross_Sectional_Analysis.ipynb`

## Installation

Install the required dependencies using `pip`:

```bash
pip install abc collections gmsh matplotlib mpl_toolkits numba numpy pandas scipy tabulate tqdm
```

## License

This software was developed as part of a Master Thesis at ETH Zurich at the [Chair of Structural Mechanics and Monitoring](https://chatzi.ibk.ethz.ch/).
