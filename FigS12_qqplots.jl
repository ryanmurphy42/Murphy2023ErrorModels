# Murphy et al. (2023) 
# Implementing measurement error models with mechanistic mathematical models
# in a likelihood-based framework for estimation, identifiability analysis, 
# and prediction in the life sciences

# FigS12_qqplots

# Structure
##### 1 - noise = additive Gaussian; MLE = normal
##### 2 - noise = additive Gaussian; MLE = lognormal
##### 3 - noise = additive Gaussian; MLE = Poisson # not applicable
##### 4 - noise = lognormal; MLE = normal # see Fig2_ODELinear_LogNormalNoise_NormalFit
##### 5 - noise = lognormal; MLE = lognormal # see Fig4_ODELinear_LogNormalNoise_LogNormalFit
##### 6 - noise = lognormal; MLE = Poisson # not applicable
##### 9 - noise = Poisson; MLE = Poisson
##### 7 - noise = Poisson; MLE = normal
##### 8 - noise = Poisson; MLE = lognormal
##### 12 - noise = Poisson; MLE = Poisson
##### 10 - noise = Poisson; MLE = normal
##### 11 - noise = Poisson; MLE = lognormal

#######################################################################################
## Initialisation including packages, plot options, filepaths to save outputs

using Plots
using LinearAlgebra
using NLopt
using .Threads 
using Interpolations
using Distributions
using Roots
using LaTeXStrings
using CSV
using DataFrames
using DifferentialEquations
using Random
using StatsPlots

pyplot() # plot options
fnt = Plots.font("sans-serif", 28) # plot options
cur_colors = palette(:default) # plot options
isdir(pwd() * "\\FigS12_qqplots_new\\") || mkdir(pwd() * "\\FigS12_qqplots_new") # make folder to save figures if doesnt already exist
filepath_save = [pwd() * "\\FigS12_qqplots_new\\"] # location to save figures

colour1=:green2
colour2=:magenta

#########################################################################
#### Mathematical model

a = zeros(3);
r1=1.0;
r2=0.5;
C10=100.0;
C20=10.0;
t_max = 2.0; # for additive Gaussian, Poisson
t=LinRange(0.0,t_max, 31); # synthetic data measurement times
t_smooth =  LinRange(0.0,t_max, 101); # for plotting known values and best-fits

# Solving the mathematical model.
function DE!(du,u,p,t)
    r1,r2=p
    du[1]=-r1*u[1];
    du[2]=r1*u[1] - r2*u[2];
end

function odesolver(t,r1,r2,C0)
    p=[r1,r2]
    tspan=(0.0,maximum(t));
    prob=ODEProblem(DE!,C0,tspan,p);
    sol=solve(prob,saveat=t);
    return sol[1:2,:];
end
    
function model(t,a)
    y=zeros(length(t))
    y=odesolver(t,a[1],a[2],[C10,C20])
    return y
end

#########################################################################
#### Measurement error model

#########################################################################
##### 1 - noise = additive Gaussian; MLE = normal

Sd_normal = 5.0; # generate additive Gaussian data 

Random.seed!(1) # set random seed
data0=0.0*zeros(2,length(t)); # generate vector to data from known parameters
data=0.0*zeros(2,length(t)); # generate vector to data from known parameters + noise
data0=model(t,[r1,r2]); # generate data using known parameter values to add noise
data0_smooth = model(t_smooth,[r1,r2]); # generate data using known parameter values for plotting
data = [rand(Normal(mi,Sd_normal)) for mi in model(t,[r1,r2])]; # generate the noisy data

df_data_normal = DataFrame(c1 = data[1,:], c2=data[2,:])
CSV.write(filepath_save[1] * "Data_normal.csv", df_data_normal)
    
 # initial guesses for MLE search
 rr1=0.95;
 rr2=0.43;
 Sdd=2.0;
 
 function error(data,a)
     y=zeros(length(t))
     y=model(t,a);
     e=0;
     data_dists=[Normal(mi,a[3]) for mi in model(t,a[1:2])]; # Normal noise model
     e+=sum([loglikelihood(data_dists[i],data[i]) for i in 1:length(data_dists)])
     return e
 end
 
 function fun(a)
     return error(data,a)
 end

 function optimise(fun,θ₀,lb,ub) 
     tomax = (θ,∂θ) -> fun(θ)
     opt = Opt(:LN_NELDERMEAD,length(θ₀))
     opt.max_objective = tomax
     opt.lower_bounds = lb       # Lower bound
     opt.upper_bounds = ub       # Upper bound
     opt.maxtime = 30.0; # maximum time in seconds
     res = optimize(opt,θ₀)
     return res[[2,1]]
 end

 # MLE

 θG = [rr1,rr2,Sdd] # first guess

 # lower and upper bounds for parameter estimation
 r1_lb = 0.5
 r1_ub = 1.5
 r2_lb = 0.1
 r2_ub = 0.9
 Sd_lb = 0.1;
 Sd_ub = 20.0;
 
 lb=[r1_lb,r2_lb,Sd_lb];
 ub=[r1_ub,r2_ub,Sd_ub];

 # MLE optimisation
 (xopt,fopt)  = optimise(fun,θG,lb,ub)

 # storing MLE 
    fmle=fopt
    r1mle=xopt[1]
    r2mle=xopt[2]
    Sdmle=xopt[3]

 data0_smooth_MLE = model(t_smooth,[r1mle,r2mle]); # generate data using known parameter values for plotting

 # plot model simulated at MLE and data
 p1=plot(t_smooth,data0_smooth_MLE[1,:],lw=3,linecolor=colour1) # solid - c1
 p1=plot!(t_smooth,data0_smooth_MLE[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
 p1=scatter!(t,data[1,:],xlims=(0,2),ylims=(0,125),legend=false,markersize = 8,markercolor=colour1,markerstrokewidth=0) # scatter c1
 p1=scatter!(t,data[2,:],xlims=(0,2.1),ylims=(0,125),legend=false,markersize = 8,markercolor=colour2,markerstrokewidth=0) # scatter c2
 display(p1)


#   residuals at MLE
  data0_MLE_recomputed_for_residuals = model(t,[r1mle,r2mle,C10,C20]); # generate data using known parameter values for plotting
  data_residuals = data-data0_MLE_recomputed_for_residuals

  # plot qq-plot
  p_residuals_qqplot_snormal_mnormal = plot(qqplot(Normal,[data_residuals[1,:];data_residuals[2,:]],lw=2,linecolor=:black,linestyle=:dot,markersize = 8,markercolor=:black),legend=false,xlab=L"\mathrm{Normal}",ylab=L"\mathrm{Residuals}",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt)
  display(p_residuals_qqplot_snormal_mnormal)
  savefig(p_residuals_qqplot_snormal_mnormal,filepath_save[1] * "Fig" * "p_residuals_qqplot_snormal_mnormal"   * ".pdf")

#########################################################################
##### 2 - noise = additive Gaussian; MLE = lognormal

# load data
df_data_normal= CSV.read([filepath_save[1] * "Data_normal.csv"],DataFrame)
data = ([transpose(df_data_normal[:,1]); transpose(df_data_normal[:,2])])

  # initial guesses for MLE
  rr1=1.2;
  rr2=0.8;
  Sdinit=0.35;

  function error(data,a)
      y=zeros(length(t))
      y=model(t,a); # simulate deterministic model with parameters a
      e=0;
      data_dists=[LogNormal(0,a[3]) for mi in y]; # LogNormal distribution
      e+=sum([loglikelihood(data_dists[i],data[i]./y[i]) for i in 1:length(data_dists)])  
      return e
  end
  
  function fun(a)
      return error(data,a)
  end

  function optimise(fun,θ₀,lb,ub) 
      tomax = (θ,∂θ) -> fun(θ)
      opt = Opt(:LN_NELDERMEAD,length(θ₀))
      opt.max_objective = tomax
      opt.lower_bounds = lb       # Lower bound
      opt.upper_bounds = ub       # Upper bound
      opt.maxtime = 30.0; # maximum time in seconds
      res = optimize(opt,θ₀)
      return res[[2,1]]
  end


  θG = [rr1,rr2,Sdinit] # first guess

    # lower and upper bounds for parameter estimation
    r1_lb = 0.5
    r1_ub = 1.5
    r2_lb = 0.1
    r2_ub = 0.9
    Sd_lb = 0.2
    Sd_ub = 0.8

    lb=[r1_lb,r2_lb,Sd_lb];
    ub=[r1_ub,r2_ub,Sd_ub];

    # MLE optimisation
    (xopt,fopt)  = optimise(fun,θG,lb,ub)

    # storing MLE 
    fmle=fopt
    r1mle=xopt[1]
    r2mle=xopt[2]
    Sdmle=xopt[3]


    data0_smooth_MLE = model(t_smooth,[r1mle,r2mle]); # generate data using known parameter values for plotting

    # plot model simulated at MLE and data
    p1=plot(t_smooth,data0_smooth_MLE[1,:],lw=3,linecolor=colour1) # solid - c1
    p1=plot!(t_smooth,data0_smooth_MLE[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
    p1=scatter!(t,data[1,:],xlims=(0,2),ylims=(0,125),legend=false,markersize = 8,markercolor=colour1,markerstrokewidth=0) # scatter c1
    p1=scatter!(t,data[2,:],xlims=(0,2.1),ylims=(0,125),legend=false,markersize = 8,markercolor=colour2,markerstrokewidth=0) # scatter c2
    display(p1)

     # plot residuals
     t_residuals = t;
     data0_MLE_recomputed_for_residuals = model(t,[r1mle,r2mle]); # generate data using known parameter values for plotting
     data_residuals = data0_MLE_recomputed_for_residuals-data
     maximum(abs.(data_residuals))
     data_residuals_multiplicative = data./data0_MLE_recomputed_for_residuals
     maximum(data_residuals_multiplicative)

     # plot qq-plot lognormal
     p_residuals_qqplot_snormal_mlognormal = plot(qqplot(LogNormal,[data_residuals_multiplicative[1,:];data_residuals_multiplicative[2,:]],lw=2,linecolor=:black,linestyle=:dot,markersize = 8,markercolor=:black),legend=false,xlab=L"\mathrm{LogNormal}",ylab=L"\hat{e}_{i}",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt)
     display(p_residuals_qqplot_snormal_mlognormal)
     savefig(p_residuals_qqplot_snormal_mlognormal,filepath_save[1] * "Figp_residuals_qqplot_snormal_mlognormal"   * ".pdf")
   


#########################################################################
##### 3 - noise = additive Gaussian; MLE = Poisson

# not applicable

#########################################################################
##### 4 - noise = lognormal; MLE = normal

# see Fig2_ODELinear_LogNormalNoise_NormalFit 

#########################################################################
##### 5 - noise = lognormal; MLE = lognormal

# see Fig4_ODELinear_LogNormalNoise_LogNormalFit

#########################################################################
##### 6 - noise = lognormal; MLE = Poisson

# not applicable

#########################################################################
##### 9 - noise = Poisson; MLE = Poisson

# generate Poisson data 

Random.seed!(1) # set random seed

data0=0.0*zeros(2,length(t)); # generate vector to data from known parameters
data=0.0*zeros(2,length(t)); # generate vector to data from known parameters + noise

data0=model(t,[r1,r2]); # generate data using known parameter values to add noise
data0_smooth = model(t_smooth,[r1,r2]); # generate data using known parameter values for plotting

data = [rand(Poisson(mi)) for mi in model(t,[r1,r2])]; # generate the noisy data

df_data_Poisson = DataFrame(c1 = data[1,:], c2=data[2,:])
CSV.write(filepath_save[1] * "Data_Poisson.csv", df_data_Poisson)

    
 # initial guesses for MLE search
 rr1=0.95;
 rr2=0.43;
 
 function error(data,a)
     y=zeros(length(t))
     y=model(t,a);
     e=0;
     data_dists=[Poisson(mi) for mi in model(t,a[1:2])]; # Normal noise model
     e+=sum([loglikelihood(data_dists[i],data[i]) for i in 1:length(data_dists)])
     return e
 end
 
 function fun(a)
     return error(data,a)
 end

 function optimise(fun,θ₀,lb,ub) 
     tomax = (θ,∂θ) -> fun(θ)
     opt = Opt(:LN_NELDERMEAD,length(θ₀))
     opt.max_objective = tomax
     opt.lower_bounds = lb       # Lower bound
     opt.upper_bounds = ub       # Upper bound
     opt.maxtime = 30.0; # maximum time in seconds
     res = optimize(opt,θ₀)
     return res[[2,1]]
 end

 # MLE

 θG = [rr1,rr2] # first guess

 # lower and upper bounds for parameter estimation
 r1_lb = 0.5
 r1_ub = 1.5
 r2_lb = 0.1
 r2_ub = 0.9
 
 lb=[r1_lb,r2_lb];
 ub=[r1_ub,r2_ub];

 # MLE optimisation
 (xopt,fopt)  = optimise(fun,θG,lb,ub)

 # storing MLE 
 fmle=fopt
 r1mle=xopt[1]
 r2mle=xopt[2]


 data0_smooth_MLE = model(t_smooth,[r1mle,r2mle]); # generate data using known parameter values for plotting


 # plot model simulated at MLE and data
 p1=plot(t_smooth,data0_smooth_MLE[1,:],lw=3,linecolor=colour1) # solid - c1
 p1=plot!(t_smooth,data0_smooth_MLE[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
 p1=scatter!(t,data[1,:],xlims=(0,2),ylims=(0,125),legend=false,markersize = 8,markercolor=colour1,markerstrokewidth=0) # scatter c1
 p1=scatter!(t,data[2,:],xlims=(0,2.1),ylims=(0,125),legend=false,markersize = 8,markercolor=colour2,markerstrokewidth=0) # scatter c2
 display(p1)


  # plot qq-plot
  p_residuals_qqplot_spoisson_mpoisson = plot(qqplot(Poisson,[data[1,:];data[2,:]],lw=2,linecolor=:black,linestyle=:dot,markersize = 8,markercolor=:black),legend=false,xlab=L"\mathrm{Poisson \ (one \ rate)}",ylab=L"\hat{e}_{i}",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt)
  display(p_residuals_qqplot_spoisson_mpoisson)
  savefig(p_residuals_qqplot_spoisson_mpoisson,filepath_save[1] * "Figp_residuals_qqplot_spoisson_mpoisson"   * ".pdf")



#########################################################################
##### 7 - noise = Poisson; MLE = normal


# load data
df_data_Poisson = CSV.read([filepath_save[1] * "Data_poisson.csv"],DataFrame)
data = ([transpose(df_data_Poisson[:,1]); transpose(df_data_Poisson[:,2])])

 # initial guesses for MLE search
 rr1=0.95;
 rr2=0.43;
 Sdinit=2.0;

 function error(data,a)
     y=zeros(length(t))
     y=model(t,a);
     e=0;
     data_dists=[Normal(mi,a[3]) for mi in model(t,a[1:2])]; # Normal noise model
     e+=sum([loglikelihood(data_dists[i],data[i]) for i in 1:length(data_dists)])
     return e
 end
 
 function fun(a)
     return error(data,a)
 end

 function optimise(fun,θ₀,lb,ub) 
     tomax = (θ,∂θ) -> fun(θ)
     opt = Opt(:LN_NELDERMEAD,length(θ₀))
     opt.max_objective = tomax
     opt.lower_bounds = lb       # Lower bound
     opt.upper_bounds = ub       # Upper bound
     opt.maxtime = 30.0; # maximum time in seconds
     res = optimize(opt,θ₀)
     return res[[2,1]]
 end

 # MLE

 θG = [rr1,rr2,Sdinit] # first guess

 # lower and upper bounds for parameter estimation
 r1_lb = 0.5
 r1_ub = 1.5
 r2_lb = 0.1
 r2_ub = 0.9
 Sd_lb = 0.1;
 Sd_ub = 20.0;
 
 lb=[r1_lb,r2_lb,Sd_lb];
 ub=[r1_ub,r2_ub,Sd_ub];

 # MLE optimisation
 (xopt,fopt)  = optimise(fun,θG,lb,ub)

 # storing MLE 
 fmle=fopt
 r1mle=xopt[1]
 r2mle=xopt[2]
 Sdmle=xopt[3]

 data0_smooth_MLE = model(t_smooth,[r1mle,r2mle]); # generate data using known parameter values for plotting

 # plot model simulated at MLE and data
 p1=plot(t_smooth,data0_smooth_MLE[1,:],lw=3,linecolor=colour1) # solid - c1
 p1=plot!(t_smooth,data0_smooth_MLE[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
 p1=scatter!(t,data[1,:],xlims=(0,2),ylims=(0,125),legend=false,markersize = 8,markercolor=colour1,markerstrokewidth=0) # scatter c1
 p1=scatter!(t,data[2,:],xlims=(0,2.1),ylims=(0,125),legend=false,markersize = 8,markercolor=colour2,markerstrokewidth=0) # scatter c2
 display(p1)


#  plot residuals at MLE
  data0_MLE_recomputed_for_residuals = model(t,[r1mle,r2mle,C10,C20]); # generate data using known parameter values for plotting
  data_residuals = data-data0_MLE_recomputed_for_residuals


  # plot qq-plot
  p_residuals_qqplot_spoisson_mnormal = plot(qqplot(Normal,[data_residuals[1,:];data_residuals[2,:]],lw=2,linecolor=:black,linestyle=:dot,markersize = 8,markercolor=:black),legend=false,xlab=L"\mathrm{Normal}",ylab=L"\mathrm{Residuals}",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt)
  display(p_residuals_qqplot_spoisson_mnormal)
  savefig(p_residuals_qqplot_spoisson_mnormal,filepath_save[1] * "Fig" * "p_residuals_qqplot_spoisson_mnormal"   * ".pdf")




#########################################################################
##### 8 - noise = Poisson; MLE = lognormal


# load data
df_data_Poisson = CSV.read([filepath_save[1] * "Data_poisson.csv"],DataFrame)
data = ([transpose(df_data_Poisson[:,1]); transpose(df_data_Poisson[:,2])])

  # initial guesses for MLE
  rr1=1.2;
  rr2=0.8;
  Sdinit=0.35;

  TH=-1.921; #95% confidence interval threshold (for confidence interval for model parameters, and confidence set for model solutions)
  TH_realisations = -2.51 # 97.5 confidence interval threshold (for model realisations)
  THalpha = 0.025; # 97.5% data realisations interval threshold

  function error(data,a)
      y=zeros(length(t))
      y=model(t,a); # simulate deterministic model with parameters a
      e=0;
      data_dists=[LogNormal(0,a[3]) for mi in y]; # LogNormal distribution
      e+=sum([loglikelihood(data_dists[i],data[i]./y[i]) for i in 1:length(data_dists)])  
      return e
  end
  
  function fun(a)
      return error(data,a)
  end

  function optimise(fun,θ₀,lb,ub) 
      tomax = (θ,∂θ) -> fun(θ)
      opt = Opt(:LN_NELDERMEAD,length(θ₀))
      opt.max_objective = tomax
      opt.lower_bounds = lb       # Lower bound
      opt.upper_bounds = ub       # Upper bound
      opt.maxtime = 30.0; # maximum time in seconds
      res = optimize(opt,θ₀)
      return res[[2,1]]
  end


  θG = [rr1,rr2,Sdinit] # first guess

    # lower and upper bounds for parameter estimation
    r1_lb = 0.5
    r1_ub = 1.5
    r2_lb = 0.1
    r2_ub = 0.9
    Sd_lb = 0.2
    Sd_ub = 0.8

    lb=[r1_lb,r2_lb,Sd_lb];
    ub=[r1_ub,r2_ub,Sd_ub];

    # MLE optimisation
    (xopt,fopt)  = optimise(fun,θG,lb,ub)

    # storing MLE 
    fmle=fopt
    r1mle=xopt[1]
    r2mle=xopt[2]
    Sdmle=xopt[3]

    data0_smooth_MLE = model(t_smooth,[r1mle,r2mle]); # generate data using known parameter values for plotting

    # plot model simulated at MLE and data
    p1=plot(t_smooth,data0_smooth_MLE[1,:],lw=3,linecolor=colour1) # solid - c1
    p1=plot!(t_smooth,data0_smooth_MLE[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
    p1=scatter!(t,data[1,:],xlims=(0,2),ylims=(0,125),legend=false,markersize = 8,markercolor=colour1,markerstrokewidth=0) # scatter c1
    p1=scatter!(t,data[2,:],xlims=(0,2.1),ylims=(0,125),legend=false,markersize = 8,markercolor=colour2,markerstrokewidth=0) # scatter c2
    display(p1)

     # plot residuals
     t_residuals = t;
     data0_MLE_recomputed_for_residuals = model(t,[r1mle,r2mle]); # generate data using known parameter values for plotting
     data_residuals = data0_MLE_recomputed_for_residuals-data
     maximum(abs.(data_residuals))
     data_residuals_multiplicative = data./data0_MLE_recomputed_for_residuals
     maximum(data_residuals_multiplicative)


     # plot qq-plot lognormal
     p_residuals_qqplot_spoisson_mlognormal = plot(qqplot(LogNormal,[data_residuals_multiplicative[1,:];data_residuals_multiplicative[2,:]],lw=2,linecolor=:black,linestyle=:dot,markersize = 8,markercolor=:black),legend=false,xlab=L"\mathrm{LogNormal}",ylab=L"\hat{e}_{i}",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt)
     display(p_residuals_qqplot_spoisson_mlognormal)
     savefig(p_residuals_qqplot_spoisson_mlognormal,filepath_save[1] * "Figp_residuals_qqplot_spoisson_mlognormal"   * ".pdf")
   


#########################################################################
##### 12 - noise = Poisson; MLE = Poisson

# generate Poisson data 

Random.seed!(1) # set random seed

data0=0.0*zeros(2,length(t)); # generate vector to data from known parameters
data=0.0*zeros(2,length(t)); # generate vector to data from known parameters + noise

data0=model(t,[r1,r2]); # generate data using known parameter values to add noise
data0_smooth = model(t_smooth,[r1,r2]); # generate data using known parameter values for plotting

data = [rand(Poisson(mi)) for mi in model(t,[r1,r2])]; # generate the noisy data

df_data_Poisson2 = DataFrame(c1 = data[1,:], c2=data[2,:])
CSV.write(filepath_save[1] * "Data_Poisson2.csv", df_data_Poisson2)

 # initial guesses for MLE search
 rr1=0.95;
 rr2=0.43;

 function error(data,a)
     y=zeros(length(t))
     y=model(t,a);
     e=0;
     data_dists=[Poisson(mi) for mi in model(t,a[1:2])]; # Normal noise model
     e+=sum([loglikelihood(data_dists[i],data[i]) for i in 1:length(data_dists)])
     return e
 end
 
 function fun(a)
     return error(data,a)
 end

 function optimise(fun,θ₀,lb,ub) 
     tomax = (θ,∂θ) -> fun(θ)
     opt = Opt(:LN_NELDERMEAD,length(θ₀))
     opt.max_objective = tomax
     opt.lower_bounds = lb       # Lower bound
     opt.upper_bounds = ub       # Upper bound
     opt.maxtime = 30.0; # maximum time in seconds
     res = optimize(opt,θ₀)
     return res[[2,1]]
 end

 # MLE

 θG = [rr1,rr2] # first guess

 # lower and upper bounds for parameter estimation
 r1_lb = 0.5
 r1_ub = 1.5
 r2_lb = 0.1
 r2_ub = 0.9
 
 lb=[r1_lb,r2_lb];
 ub=[r1_ub,r2_ub];

 # MLE optimisation
 (xopt,fopt)  = optimise(fun,θG,lb,ub)

 # storing MLE 
 fmle=fopt
 r1mle=xopt[1]
 r2mle=xopt[2]


 data0_smooth_MLE = model(t_smooth,[r1mle,r2mle]); # generate data using known parameter values for plotting


 # plot model simulated at MLE and data
 p1=plot(t_smooth,data0_smooth_MLE[1,:],lw=3,linecolor=colour1) # solid - c1
 p1=plot!(t_smooth,data0_smooth_MLE[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
 p1=scatter!(t,data[1,:],xlims=(0,2),ylims=(0,125),legend=false,markersize = 8,markercolor=colour1,markerstrokewidth=0) # scatter c1
 p1=scatter!(t,data[2,:],xlims=(0,2.1),ylims=(0,125),legend=false,markersize = 8,markercolor=colour2,markerstrokewidth=0) # scatter c2
 display(p1)


  # plot qq-plot
  p_residuals_qqplot_spoisson_mpoisson2 = plot(qqplot(Poisson,[data[1,:];data[2,:]],lw=2,linecolor=:black,linestyle=:dot,markersize = 8,markercolor=:black),legend=false,xlab=L"\mathrm{Poisson \ (one \ rate)}",ylab=L"\hat{e}_{i}",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt)
  display(p_residuals_qqplot_spoisson_mpoisson2)
  savefig(p_residuals_qqplot_spoisson_mpoisson2,filepath_save[1] * "Figp_residuals_qqplot_spoisson_mpoisson2"   * ".pdf")

#########################################################################
##### 10 - noise = Poisson; MLE = normal


# load data
df_data_Poisson2 = CSV.read([filepath_save[1] * "Data_poisson2.csv"],DataFrame)
data = ([transpose(df_data_Poisson2[:,1]); transpose(df_data_Poisson2[:,2])])

    
 # initial guesses for MLE search
 rr1=0.95;
 rr2=0.43;
 Sdinit=2.0;

 function error(data,a)
     y=zeros(length(t))
     y=model(t,a);
     e=0;
     data_dists=[Normal(mi,a[3]) for mi in model(t,a[1:2])]; # Normal noise model
     e+=sum([loglikelihood(data_dists[i],data[i]) for i in 1:length(data_dists)])
     return e
 end
 
 function fun(a)
     return error(data,a)
 end

 function optimise(fun,θ₀,lb,ub) 
     tomax = (θ,∂θ) -> fun(θ)
     opt = Opt(:LN_NELDERMEAD,length(θ₀))
     opt.max_objective = tomax
     opt.lower_bounds = lb       # Lower bound
     opt.upper_bounds = ub       # Upper bound
     opt.maxtime = 30.0; # maximum time in seconds
     res = optimize(opt,θ₀)
     return res[[2,1]]
 end

 # MLE

 θG = [rr1,rr2,Sdinit] # first guess

 # lower and upper bounds for parameter estimation
 r1_lb = 0.5
 r1_ub = 1.5
 r2_lb = 0.1
 r2_ub = 0.9
 Sd_lb = 0.1;
 Sd_ub = 20.0;
 
 lb=[r1_lb,r2_lb,Sd_lb];
 ub=[r1_ub,r2_ub,Sd_ub];

 # MLE optimisation
 (xopt,fopt)  = optimise(fun,θG,lb,ub)

 # storing MLE 
 fmle=fopt
 r1mle=xopt[1]
 r2mle=xopt[2]
 Sdmle=xopt[3]


 data0_smooth_MLE = model(t_smooth,[r1mle,r2mle]); # generate data using known parameter values for plotting


 # plot model simulated at MLE and data
 p1=plot(t_smooth,data0_smooth_MLE[1,:],lw=3,linecolor=colour1) # solid - c1
 p1=plot!(t_smooth,data0_smooth_MLE[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
 p1=scatter!(t,data[1,:],xlims=(0,2),ylims=(0,125),legend=false,markersize = 8,markercolor=colour1,markerstrokewidth=0) # scatter c1
 p1=scatter!(t,data[2,:],xlims=(0,2.1),ylims=(0,125),legend=false,markersize = 8,markercolor=colour2,markerstrokewidth=0) # scatter c2
 display(p1)


# plot residuals at MLE
  data0_MLE_recomputed_for_residuals = model(t,[r1mle,r2mle,C10,C20]); # generate data using known parameter values for plotting
  data_residuals = data-data0_MLE_recomputed_for_residuals


  # plot qq-plot
  p_residuals_qqplot_spoisson_mnormal2 = plot(qqplot(Normal,[data_residuals[1,:];data_residuals[2,:]],lw=2,linecolor=:black,linestyle=:dot,markersize = 8,markercolor=:black),legend=false,xlab=L"\mathrm{Normal}",ylab=L"\mathrm{Residuals}",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt)
  display(p_residuals_qqplot_spoisson_mnormal2)
  savefig(p_residuals_qqplot_spoisson_mnormal2,filepath_save[1] * "Fig" * "p_residuals_qqplot_spoisson_mnormal2"   * ".pdf")


#########################################################################
##### 11 - noise = Poisson; MLE = lognormal

# load data
df_data_Poisson2 = CSV.read([filepath_save[1] * "Data_poisson2.csv"],DataFrame)
data = ([transpose(df_data_Poisson2[:,1]); transpose(df_data_Poisson2[:,2])])

  # initial guesses for MLE
  rr1=1.2;
  rr2=0.8;
  Sdinit=0.35;

  function error(data,a)
      y=zeros(length(t))
      y=model(t,a); # simulate deterministic model with parameters a
      e=0;
      data_dists=[LogNormal(0,a[3]) for mi in y]; # LogNormal distribution
      e+=sum([loglikelihood(data_dists[i],data[i]./y[i]) for i in 1:length(data_dists)])  
      return e
  end
  
  function fun(a)
      return error(data,a)
  end

  function optimise(fun,θ₀,lb,ub) 
      tomax = (θ,∂θ) -> fun(θ)
      opt = Opt(:LN_NELDERMEAD,length(θ₀))
      opt.max_objective = tomax
      opt.lower_bounds = lb       # Lower bound
      opt.upper_bounds = ub       # Upper bound
      opt.maxtime = 30.0; # maximum time in seconds
      res = optimize(opt,θ₀)
      return res[[2,1]]
  end


  θG = [rr1,rr2,Sdinit] # first guess

    # lower and upper bounds for parameter estimation
    r1_lb = 0.5
    r1_ub = 1.5
    r2_lb = 0.1
    r2_ub = 0.9
    Sd_lb = 0.2
    Sd_ub = 0.8

    lb=[r1_lb,r2_lb,Sd_lb];
    ub=[r1_ub,r2_ub,Sd_ub];

    # MLE optimisation
    (xopt,fopt)  = optimise(fun,θG,lb,ub)

    # storing MLE 
    fmle=fopt
    r1mle=xopt[1]
    r2mle=xopt[2]
    Sdmle=xopt[3]

    data0_smooth_MLE = model(t_smooth,[r1mle,r2mle]); # generate data using known parameter values for plotting

    # plot model simulated at MLE and data
    p1=plot(t_smooth,data0_smooth_MLE[1,:],lw=3,linecolor=colour1) # solid - c1
    p1=plot!(t_smooth,data0_smooth_MLE[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
    p1=scatter!(t,data[1,:],xlims=(0,2),ylims=(0,125),legend=false,markersize = 8,markercolor=colour1,markerstrokewidth=0) # scatter c1
    p1=scatter!(t,data[2,:],xlims=(0,2.1),ylims=(0,125),legend=false,markersize = 8,markercolor=colour2,markerstrokewidth=0) # scatter c2
    display(p1)

     # plot residuals
     t_residuals = t;
     data0_MLE_recomputed_for_residuals = model(t,[r1mle,r2mle]); # generate data using known parameter values for plotting
     data_residuals = data0_MLE_recomputed_for_residuals-data
     maximum(abs.(data_residuals))
     data_residuals_multiplicative = data./data0_MLE_recomputed_for_residuals
     maximum(data_residuals_multiplicative)

  
     # plot qq-plot lognormal
     p_residuals_qqplot_spoisson_mlognormal2 = plot(qqplot(LogNormal,[data_residuals_multiplicative[1,:];data_residuals_multiplicative[2,:]],lw=2,linecolor=:black,linestyle=:dot,markersize = 8,markercolor=:black),legend=false,xlab=L"\mathrm{LogNormal}",ylab=L"\hat{e}_{i}",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt)
     display(p_residuals_qqplot_spoisson_mlognormal2)
     savefig(p_residuals_qqplot_spoisson_mlognormal2,filepath_save[1] * "Figp_residuals_qqplot_spoisson_mlognormal2"   * ".pdf")
   
