# Murphy et al. (2023)  Implementing measurement error models in a likelihood-based framework for estimation, identifiability analysis, and prediction in the life sciences

Preprint coming soon to arXiv

This repository holds key Julia code used to generate figures in the the manuscript.

Please contact Ryan Murphy for any queries or questions.

Code developed and run in July 2023 using:

- Julia Version  1.7.2 (see https://julialang.org/downloads/ )
- Julia packages: Plots, LinearAlgebra, NLopt, .Threads, Interpolations, Distributions, Roots, LaTeXStrings, DifferentialEquations, CSV, DataFrames, Parsers, MethodOfLines, Random, StatsPlots, StructuralIdentifiability, SpecialFunctions.

## Guide to using the code
The script InstallPackages.jl can be used to install packages (by uncommenting the relevant lines). There scripts are summarised in the table below.


| | Script        | Figures in manuscript | Short description           | 
| :---:   | :---: | :---: | :---: |
|1| Fig1.jl  | Figure 1 | ODELinear_NormalNoise_NormalFit |
|2| Fig2.jl     | Figures 2-3 |  ODELinear_LogNormalNoise_NormalFit   |  
|3|  Fig4.jl | Figure 4 | ODELinear_LogNormalNoise_LogNormalFit  |  
|4| Fig5.jl  | Figure 5 | ODENonlinear_PoissonNoise_PoissonFit  |  
|5| Fig6.jl | Figure 6  | PDE_PoissonNoise_PoissonFit | 
|6| Fig7.jl | Figures 7-11 | Coverage_ODELinear_NormalNoise_NormalFit_MLE  | 
|7| FigS1.jl | Supplementary Figure S1 | PDE_ComparisonofAnalyticalandNumericalSolutions  | 
|8| FigS2.jl | Supplementary Figure S2-3 | FullLikelhood_ODELinear_NormalNoise_NormalFit   | 
|9| FigS4.jl | Supplementary Figure S4 |  Coverage_ODELinear_NormalNoise_NormalFit_FullLikelihood | 
|10| FigS5.jl |Supplementary Figure S5 | ODElotkavolterra_PoissonNoise_PoissonFit    | 
|11| FigS6.jl |Supplementary Figure S6 | ODElotkavolterra_PoissonNoise_NormalFit   | 
|12| FigS7.jl |Supplementary Figure S7 |  DifferenceEquation_PoissonNoise_PoissonFit  | 
|13| FigS8.jl | Supplementary Figures S8-10 | Coverage_ODELinear_LogNormalNoise_LogNormalFit_MLE  | 
|14| FigS11.jl |Supplementary Figure S11 |Coverage_ODELinear_LogNormalNoise_LogNormalFit_FullLikelihood   | 


