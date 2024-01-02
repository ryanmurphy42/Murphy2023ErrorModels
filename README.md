# Murphy et al. (2024)  Implementing measurement error models with mechanistic mathematical models in a likelihood-based framework for estimation, identifiability analysis, and prediction in the life sciences

Preprint: https://arxiv.org/abs/2307.01539

This repository holds the key Julia code used to generate figures in the manuscript (figure numbers correspond to version 2 on arXiv).

Please contact Ryan Murphy for any queries or questions.

Code developed and run in July 2023 using:

- Julia Version  1.7.2 (see https://julialang.org/downloads/ )
- Julia packages: Plots, LinearAlgebra, NLopt, .Threads, Interpolations, Distributions, Roots, LaTeXStrings, DifferentialEquations, CSV, DataFrames, Parsers, MethodOfLines, Random, StatsPlots, StructuralIdentifiability, SpecialFunctions.

## Guide to using the code
The script InstallPackages.jl can be used to install packages (by uncommenting the relevant lines). Scripts are summarised in the table below.


| | Script        | Figures in manuscript | Short description           | 
| :---:   | :---: | :---: | :---: |
|1| Fig2.jl  | Figure 2 | ODELinear_NormalNoise_NormalFit |
|2| Fig3.jl     | Figures 3-4 |  ODELinear_LogNormalNoise_NormalFit   |  
|3| Fig5.jl | Figure 5 | ODELinear_LogNormalNoise_LogNormalFit  |  
|4| Fig6.jl  | Figure 6 | ODENonlinear_PoissonNoise_PoissonFit  |  
|5| Fig7.jl | Figure 7  | PDE_PoissonNoise_PoissonFit | 
|6| Fig8.jl | Figures 8-12 | Coverage_ODELinear_NormalNoise_NormalFit_MLE  | 
|7| FigS1.jl | Supplementary Figure S1 | PDE_ComparisonofAnalyticalandNumericalSolutions  | 
|8| FigS2.jl | Supplementary Figure S2-S3 | FullLikelhood_ODELinear_NormalNoise_NormalFit   | 
|9| FigS4.jl | Supplementary Figure S4 |  Coverage_ODELinear_NormalNoise_NormalFit_FullLikelihood | 
|10| FigS5.jl |Supplementary Figure S5 | ODElotkavolterra_PoissonNoise_PoissonFit    | 
|11| FigS6.jl |Supplementary Figure S6 | ODElotkavolterra_PoissonNoise_NormalFit   | 
|12| FigS7.jl |Supplementary Figure S7 |  DifferenceEquation_PoissonNoise_PoissonFit  | 
|13| FigS8.jl | Supplementary Figures S8-S10 | Coverage_ODELinear_LogNormalNoise_LogNormalFit_MLE  | 
|14| FigS11.jl |Supplementary Figure S11 |Coverage_ODELinear_LogNormalNoise_LogNormalFit_FullLikelihood   | 
|15| FigS12.jl |Supplementary Figure S12 |QQplots  | 
