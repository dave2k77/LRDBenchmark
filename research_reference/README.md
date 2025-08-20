# Research Reference

Add research papers, notes, and references related to models, estimators, and validation. Keep this updated as the project evolves.

@article{csanady2024parameter,
  title={Parameter Estimation of Long Memory Stochastic Processes},
  author={Csan{\'a}dy, B{\'a}lint and Nagy, L{\'o}r{\'a}nt and Boros, D{\'a}niel and Ivkovic, Iv{\'a}n and Kov{\'a}cs, D{\'a}vid and T{\'o}th-Lakits, Dalma and M{\'a}rkus, L{\'a}szl{\'o} and Luk{\'a}cs, Andr{\'a}s},
  journal={Frontiers in Artificial Intelligence and Applications},
  volume={381},
  pages={2548--2559},
  year={2024},
  publisher={IOS Press},
  doi={10.3233/FAIA240784},
  note={Open Access, CC BY-NC 4.0}
}

@inproceedings{Nakagawa2024LfNet,
  title = {Lf-Net: Generating Fractional Time-Series with Latent Fractional-Net},
  author = {Kei Nakagawa and Kohei Hayashi},
  booktitle = {Proceedings of the 2024 International Joint Conference on Neural Networks (IJCNN)},
  year = {2024},
  organization = {IEEE},
  pages = {To appear},
  doi = {10.1109/IJCNN60899.2024.10650271},
  note = {Available online},
  abstract = {In this paper, we introduce a novel method for generating fractional time series through the utilization of neural networks. Although Neural Stochastic Differential Equations (Neural SDEs) have been presented as a method that combines Deep Neural Networks with numerical solvers of differential equations, these typically presume the noise structure of standard Brownian motion (Bm). Contrarily, numerous real-world time series data exhibit a fractal property, characterized by a Hurst index (H) that ranges from 0 to 1. This type of fractional time series pervades various domains including physics, biology, hydrology, network research, and financial mathematics. We propose a Latent Fractional Net (Lf-Net), devised to encapsulate both the long-range dependence (H > 1/2) and roughness (H < 1/2) intrinsic to fractional time series. This is accomplished by augmenting the noise term of the Neural SDEs using fractional Brownian motion (fBm) with an arbitrary Hurst index. We prove the existence and uniqueness of the solutions of the Lf-Net and theoretically show the convergence of the numerical solutions. We demonstrate the robustness of the Lf-Net under proper nonlinear transformations and construct a generative model for time-series data. The experiments show that the calibrated generator of the model can replicate the distributional properties of the original time series, especially the Hurst index.}
}

@article{Hayashi2022fSDENet,
  title = {fSDE-Net: Generating Time Series Data with Long-term Memory},
  author = {Kohei Hayashi and Kei Nakagawa},
  journal = {arXiv preprint arXiv:2201.05974},
  year = {2022},
  url = {https://arxiv.org/abs/2201.05974},
  note = {Version 2, cs.LG, 24 Aug 2022},
  abstract = {We propose fSDE-Net, a neural fractional Stochastic Differential Equation Network, which generalizes the neural SDE-Net by using fractional Brownian motion with a Hurst index larger than half, exhibiting long-range dependence (LRD) property. We derive a numerical solver for fSDE-Net and theoretically analyze the existence and uniqueness of its solutions. Experiments on artificial and real time-series data demonstrate that fSDE-Net replicates distributional properties, especially LRD.}
}

@article{guerra2008stochastic,
  author = {Guerra, João and Nualart, David},
  title = {Stochastic differential equations driven by fractional Brownian motion and standard Brownian motion},
  journal = {Stochastics and Dynamics},
  volume = {8},
  number = {4},
  pages = {609--641},
  year = {2008},
  doi = {10.1142/S0219493708002494},
}

@article{jien2009stochastic,
  author = {Jien, Yu-Juan and Ma, Jin},
  title = {Stochastic differential equations driven by fractional Brownian motions},
  journal = {Bernoulli},
  volume = {15},
  number = {3},
  pages = {846--870},
  year = {2009},
  doi = {10.3150/08-BEJ169},
}