# Murphy et al. (2023) 
# Implementing measurement error models with mechanistic mathematical models
# in a likelihood-based framework for estimation, identifiability analysis, 
# and prediction in the life sciences

# FigS4_Coverage_ODELinear_NormalNoise_NormalFit_FullLikelihood

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
using Dates

pyplot() # plot options
fnt = Plots.font("sans-serif", 20) # plot options
global cur_colors = palette(:default) # plot options
isdir(pwd() * "\\FigS4_Coverage_ODELinear_NormalNoise_NormalFit_FullLikelihood\\") || mkdir(pwd() * "\\FigS4_Coverage_ODELinear_NormalNoise_NormalFit_FullLikelihood") # make folder to save figures if doesnt already exist
filepath_save = [pwd() * "\\FigS4_Coverage_ODELinear_NormalNoise_NormalFit_FullLikelihood\\"] # location to save figures "\\Fig2\\"] # location to save figures

#######################################################################################
## Mathematical model

a = zeros(3);
r1=1.0;
r2=0.5;
C10=100.0;
C20=25.0;
t_max = 2.0;
t=LinRange(0.0,t_max, 16); # synthetic data measurement times
t_smooth =  LinRange(0.0,t_max*1.1, 101); # for plotting known values and best-fits

Sd = 5;

function DE!(du,u,p,t)
    r1,r2=p
    du[1]=-r1*u[1];
    du[2]=r1*u[1] - r2*u[2];
end

#######################################################################################
## Solving the mathematical model

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
     

# test the coverage by repeating simulation totalseeds times
totalseeds=5000

# record if coverage true or false
r1r2_withinbounds =  zeros(totalseeds)

modelsoln_withinbounds =  zeros(totalseeds)
modelsoln_withinbounds_time =  zeros(totalseeds,2,length(t_smooth)) # realisations, number of variables, time vector

realisations_withinbounds =  zeros(totalseeds)
realisations_withinbounds_time =  zeros(totalseeds,2,length(t)) # realisations, number of variables, time vector

colour1=:green2
colour2=:magenta

println("Start: ")
println(now())

for seednum in 1:totalseeds
    println(seednum)

    #######################################################################################
    ## Generate Synthetic data


    Random.seed!(seednum) # set random seed

    data0=0.0*zeros(2,length(t)); # generate vector to data from known parameters
    data=0.0*zeros(2,length(t)); # generate vector to data from known parameters + noise
    testdata=0.0*zeros(2,length(t)); # generate vector to data from known parameters + noise

    data0=model(t,[r1,r2,C10,C20]); # generate data using known parameter values to add noise
    data0_smooth = model(t_smooth,[r1,r2,C10,C20]); # generate data using known parameter values for plotting

    # generate distributions to sample to generate noisy data
    data_dists=[Normal(mi,Sd) for mi in model(t,[r1,r2,C10,C20])]; # normal distribution - standard deviation 1
    data=[rand(data_dist) for data_dist in data_dists]; # generate the noisy data
    data[data .< 0] .= zero(eltype(data)) # set negative values to zero


    Random.seed!(seednum + 10000)
    testdata=[rand(data_dist) for data_dist in data_dists]; # generate the noisy data
    testdata[testdata .< 0] .= zero(eltype(testdata)) # set negative values to zero


    #######################################################################################
    ### Parameter identifiability analysis - MLE, profiles
        # initial guesses for MLE search
        rr1=1.01;
        rr2=0.51;
    
        TH=-3.00; #95% confidence interval threshold (2 dimensional parameter)
        THreal=-3.69; #97.5% confidence interval threshold (2 dimensional parameter)
        THalpha = 0.025; # 97.5% data realisations interval threshold

        function error(data,a)
            y=zeros(length(t))
            y=model(t,a);
            e=0;
            data_dists=[Normal(mi,Sd) for mi in model(t,a[1:2])]; # Normal noise model
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

        #######################################################################################
        # MLE

        
        θG = [rr1,rr2] # first guess
        # lower and upper bounds for parameter estimation
        r1_lb = 0.8
        r1_ub = 1.2
        r2_lb = 0.3
        r2_ub = 0.7

        lb=[r1_lb,r2_lb];
        ub=[r1_ub,r2_ub];

        # MLE optimisation
        (xopt,fopt)  = optimise(fun,θG,lb,ub)
        ytrue_smooth= model(t_smooth,[r1,r2,C10,C20]);

    #######################################################################################
        # Full likelihood-based approach

        r1min = r1_lb
        r1max = r1_ub
        r1rangefull = LinRange(r1min,r1max,201)

        r2min = r2_lb
        r2max = r2_ub
        r2rangefull = LinRange(r2min,r2max,201)

     
        # Test if true parameter value is contained within the 95% confidence region
        if (error(data,[r1rangefull[101],r2rangefull[101]]) - fopt) >= TH  # if the normalised log-likleihood at the true value is greater than the threshold
            r1r2_withinbounds[seednum] =1
        end


        # Compute the loglikelihood at all points in the domain
        fullloglikelihood= zeros(length(r1rangefull),length(r2rangefull))
        for i=1:length(r1rangefull)
            for j=1:length(r2rangefull)
                    a = [r1rangefull[i],r2rangefull[j]]
                    fullloglikelihood[i,j] = error(data,a) - fopt;
            end
        end    
    

        ### Prediction - confidence sets for the model solution
        predict_modelsolution=zeros(2,length(t_smooth),length(r1rangefull),length(r2rangefull))
        for i=1:length(r1rangefull)
            for j=1:length(r2rangefull)
                    if fullloglikelihood[i,j] > TH
                        # predict model solution
                        a = [r1rangefull[i],r2rangefull[j]]
                        predict_modelsolution[:,:,i,j]=model(t_smooth,a)        
                    end
            end
        end

        # calculate min and max model solutions from the full likelihoodd
        max_modelsolution=zeros(2,length(t_smooth))
        min_modelsolution=1000*ones(2,length(t_smooth))
        for i in 1:length(r1rangefull)
            for j in 1:length(r2rangefull)
                if fullloglikelihood[i,j] > TH
                    for kk in 1:length(t_smooth)
                        max_modelsolution[1,kk]=max(predict_modelsolution[1,kk,i,j],max_modelsolution[1,kk])
                        min_modelsolution[1,kk]=min(predict_modelsolution[1,kk,i,j],min_modelsolution[1,kk])
                        max_modelsolution[2,kk]=max(predict_modelsolution[2,kk,i,j],max_modelsolution[2,kk])
                        min_modelsolution[2,kk]=min(predict_modelsolution[2,kk,i,j],min_modelsolution[2,kk])
                    end
                end
            end
        end

        # is the true model solution within the model solution bounds?
        if (sum(ytrue_smooth .>= min_modelsolution) == 2*length(t_smooth)) .* (sum(ytrue_smooth .<= max_modelsolution) == 2*length(t_smooth))
            modelsoln_withinbounds[seednum] = 1
        end

            
        ## time
        for i in 1:length(t_smooth)
            # model solution
            if (ytrue_smooth[1,i] .>= min_modelsolution[1,i]) .* (ytrue_smooth[1,i] .<= max_modelsolution[1,i])
                modelsoln_withinbounds_time[seednum,1,i] = 1
            end
            if (ytrue_smooth[2,i] .>= min_modelsolution[2,i]) .* (ytrue_smooth[2,i] .<= max_modelsolution[2,i])
                modelsoln_withinbounds_time[seednum,2,i] = 1
            end
        end


        ### Prediction - confidence sets for the model realisation
        predict_realisations_min=zeros(2,length(t),length(r1rangefull),length(r2rangefull))
        predict_realisations_max=zeros(2,length(t),length(r1rangefull),length(r2rangefull))
        for i in 1:length(r1rangefull)
            for j in 1:length(r2rangefull)
                if fullloglikelihood[i,j] > THreal
                    a = [r1rangefull[i],r2rangefull[j]]
                    # predict data realisations
                    loop_data_dists=[Normal(mi,Sd) for mi in model(t,a)]; # normal distribution about mean
                    predict_realisations_min[:,:,i,j]= [quantile(data_dist, THalpha/2) for data_dist in loop_data_dists];                
                    predict_realisations_max[:,:,i,j]= [quantile(data_dist, 1-THalpha/2) for data_dist in loop_data_dists];         
                end
            end
        end

        # combine the confidence sets for realisations
        max_realisations=zeros(2,length(t))
        min_realisations=1000*ones(2,length(t))
        for i in 1:length(r1rangefull)
            for j in 1:length(r2rangefull)
                if fullloglikelihood[i,j] > THreal
                    for kk in 1:length(t)
                        max_realisations[1,kk]=max(predict_realisations_max[1,kk,i,j],max_realisations[1,kk])
                        min_realisations[1,kk]=min(predict_realisations_min[1,kk,i,j],min_realisations[1,kk])
                        max_realisations[2,kk]=max(predict_realisations_max[2,kk,i,j],max_realisations[2,kk])
                        min_realisations[2,kk]=min(predict_realisations_min[2,kk,i,j],min_realisations[2,kk])
                    end
                end
            end
        end

                 # is the true realisation solution within the confidence set for model realisation bounds?

        if (sum(testdata[:,2:end] .>= min_realisations[:,2:end]) == 2*(length(t)-1)) .* (sum(testdata[:,2:end] .<= max_realisations[:,2:end]) == 2*(length(t)-1))
            realisations_withinbounds[seednum] = 1
        end


        for i in 1:length(t)
            # data realisations
            if (testdata[1,i] .>= min_realisations[1,i]) .* (testdata[1,i] .<= max_realisations[1,i])
                realisations_withinbounds_time[seednum,1,i] = 1
            end
            if (testdata[2,i] .>= min_realisations[2,i]) .* (testdata[2,i] .<= max_realisations[2,i])
                realisations_withinbounds_time[seednum,2,i] = 1
            end
        end

    end

    println("End: ")
    println(now())


             
sum(r1r2_withinbounds)/totalseeds
# 0.9496


sum(modelsoln_withinbounds)/totalseeds
# 0.9538

sum(realisations_withinbounds)/totalseeds
# 0.8458


### Plots


# Model solution  - union
coverage_m_union_c1  = plot(t_smooth[2:end], sum(modelsoln_withinbounds_time[:,1,2:end],dims=1)' ./totalseeds, ylim=(0,1.1),lw=3, titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour1,legend=false,xlab=L"t",ylab=L"C_{y,0.95} \mathrm{ \ coverage}")
coverage_m_union_c1 = hline!([1.00],lw=2,linecolor=:black,linestyle=:dash)
coverage_m_union_c1 = hline!([0.95],lw=2,linecolor=:black,linestyle=:dot)
display(coverage_m_union_c1)
savefig(coverage_m_union_c1,filepath_save[1] * "Fig_" * "coverage_m_union_c1"  * ".pdf")



coverage_m_union_c2 = plot(t_smooth[2:end], sum(modelsoln_withinbounds_time[:,2,2:end],dims=1)' ./totalseeds, ylim=(0,1.1),lw=3, titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2,legend=false,xlab=L"t",ylab=L"C_{y,0.95} \mathrm{ \ coverage}")
coverage_m_union_c2 = hline!([1.00],lw=2,linecolor=:black,linestyle=:dash)
coverage_m_union_c2 = hline!([0.95],lw=2,linecolor=:black,linestyle=:dot)
display(coverage_m_union_c2)
savefig(coverage_m_union_c2,filepath_save[1] * "Fig_" * "coverage_m_union_c2"  * ".pdf")


# Data realisations - union

coverage_r_union_c1 = hline([1.00],lw=2,linecolor=:black,linestyle=:dash)
coverage_r_union_c1 = hline!([0.95],lw=2,linecolor=:black,linestyle=:dot)
coverage_r_union_c1 = scatter!(t[2:end], sum(realisations_withinbounds_time[:,1,2:end],dims=1)' ./totalseeds, ylim=(0,1.1),markersize=10,msw=0, titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,color=colour1,legend=false,xlab=L"t",ylab=L"C_{y_{i},0.95} \mathrm{ \ coverage}")
display(coverage_r_union_c1)
savefig(coverage_r_union_c1,filepath_save[1] * "Fig_" * "coverage_r_union_c1"  * ".pdf")

round(mean(sum(realisations_withinbounds_time[:,1,2:end],dims=1)' ./totalseeds),digits=3)
# 0.994

coverage_r_union_c2 = hline([1.00],lw=2,linecolor=:black,linestyle=:dash)
coverage_r_union_c2 = hline!([0.95],lw=2,linecolor=:black,linestyle=:dot)
coverage_r_union_c2 = scatter!(t[2:end], sum(realisations_withinbounds_time[:,2,2:end],dims=1)' ./totalseeds, ylim=(0,1.1),markersize=10,msw=0, titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,color=colour2,legend=false,xlab=L"t",ylab=L"C_{y_{i},0.95} \mathrm{ \ coverage}")
display(coverage_r_union_c2)
savefig(coverage_r_union_c2,filepath_save[1] * "Fig_" * "coverage_r_union_c2"  * ".pdf")

round(mean( sum(realisations_withinbounds_time[:,2,2:end],dims=1)' ./totalseeds),digits=3)
# 0.995

round(mean( [sum(realisations_withinbounds_time[:,1,2:end],dims=1)' ./totalseeds; sum(realisations_withinbounds_time[:,2,2:end],dims=1)' ./totalseeds ] ),digits=3)
# 0.994