# Murphy et al. (2023) 
# Implementing measurement error models with mechanistic mathematical models
# in a likelihood-based framework for estimation, identifiability analysis, 
# and prediction in the life sciences

# FigS2_FullLikelhood_ODELinear_NormalNoise_NormalFit

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
fnt = Plots.font("sans-serif", 20) # plot options
global mod_colors = palette(:redsblues) # plot options
global cur_colors = palette(:default) # plot options
isdir(pwd() * "\\FigS2_FullLikelhood_ODELinear_NormalNoise_NormalFit\\") || mkdir(pwd() * "\\FigS2_FullLikelhood_ODELinear_NormalNoise_NormalFit") # make folder to save figures if doesnt already exist
filepath_save = [pwd() * "\\FigS2_FullLikelhood_ODELinear_NormalNoise_NormalFit\\"] # location to save figures "\\Fig2\\"] # location to save figures

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
    
#######################################################################################
## Generate Synthetic data

    Random.seed!(1234) # set random seed

    data0=0.0*zeros(2,length(t)); # generate vector to data from known parameters
    data=0.0*zeros(2,length(t)); # generate vector to data from known parameters + noise
    data0=model(t,[r1,r2,C10,C20]); # generate data using known parameter values to add noise
    data0_smooth = model(t_smooth,[r1,r2,C10,C20]); # generate data using known parameter values for plotting
    data_dists=[Normal(mi,Sd) for mi in model(t,[r1,r2,C10,C20])];  # generate normal distributions to sample
    data=[rand(data_dist) for data_dist in data_dists]; # generate the noisy data
    data[data .< 0] .= zero(eltype(data)) # set negative values to zero

    testdata=[rand(data_dist) for data_dist in data_dists]; # generate the noisy data
    testdata[testdata .< 0] .= zero(eltype(testdata)) # set negative values to zero

#######################################################################################
### Plot data - scatter plot

xmin=-0.05;
xmax= 2.1;
ymin= 0;
ymax= 120;

colour1=:green2
colour2=:magenta

p0=plot(t_smooth,data0_smooth[1,:],lw=3,linecolor=colour1) # solid - c1
p0=plot!(t_smooth,data0_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
p0=scatter!(t,data[1,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
p0=scatter!(t,data[2,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,msw=0) # scatter c2

display(p0)
savefig(p0,filepath_save[1] * "Fig0" * ".pdf")


p0scatter=scatter(t,data[1,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 12,markercolor=colour1,markerstrokewidth=0) # scatter c1
p0scatter=scatter!(t,data[2,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 12,markercolor=colour2,xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,markerstrokewidth=0) # scatter c2

display(p0scatter)
savefig(p0scatter,filepath_save[1] * "Fig0scatter" * ".pdf")

#######################################################################################
### Parameter identifiability analysis - MLE, profiles

    # initial guesses for MLE search
    rr1=1.1;
    rr2=0.6;    
    
    # thresholds for full likelihood-based approach (two-dimensional parameter)
    TH_full_95=-3.00; #95% confidence interval  (quantile(Chisq(2), 0.95)/2)
    TH_full_975=-3.69; #97.5% confidence interval  (for model realisations) (quantile(Chisq(2), 0.975)/2)
    TH_full_alpha = 0.025; # 97.5% data realisations interval threshold

    # thresholds for profile likelihood-based approach (one-dimensional interest parameter)
    TH_prof_95=-1.92; #95% confidence interval threshold for model parameters and confidence set for model solutions (quantile(Chisq(1), 0.95)/2)
    TH_prof_975=-2.51; #97.5% confidence interval threshold (for model realisations)  (quantile(Chisq(1), 0.975)/2)
    TH_prof_alpha = 0.025; # 97.5% data realisations interval threshold


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

    # storing MLE 
    global fmle=fopt
    global r1mle=xopt[1]
    global r2mle=xopt[2]


    data0_smooth_MLE = model(t_smooth,[r1mle,r2mle,C10,C20]); # generate data using known parameter values for plotting

    # plot model simulated at MLE and save
    p1=plot(t_smooth,data0_smooth_MLE[1,:],lw=3,linecolor=colour1) # solid - c1
    p1=plot!(t_smooth,data0_smooth_MLE[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
    p1=scatter!(t,data[1,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
    p1=scatter!(t,data[2,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,msw=0) # scatter c2

    display(p1)
    savefig(p1,filepath_save[1] * "Fig1" * ".pdf")


    #######################################################################################
    #######################################################################################
    #######################################################################################
        # FULL LIKELIHOOD-BASED APPROACH

        r1min = r1_lb
        r1max = r1_ub
        r1rangefull = LinRange(r1min,r1max,201)

        r2min = r2_lb
        r2max = r2_ub
        r2rangefull = LinRange(r2min,r2max,201)

        fullloglikelihood= zeros(length(r1rangefull),length(r2rangefull))
        for i=1:length(r1rangefull)
            for j=1:length(r2rangefull)
                    a = [r1rangefull[i],r2rangefull[j]]
                    fullloglikelihood[i,j] = error(data,a) - fopt;
            end
        end


        maximum(fullloglikelihood)

        # contour plot with confidence level thresholds for df=2
        figcontourdf2 = contour(r1rangefull, r2rangefull, fullloglikelihood', levels=[-6.91,-4.61,-3.00,0],clims=(-10,0),xlabel=L"r_{1}", ylabel=L"r_{2}", fill=true,colorbar_title=L"\hat{\ell}(\theta \ | \ y_{1:I}^{o})")
        figcontourdf2 = scatter(figcontourdf2,[r1mle],[r2mle],ms=10,color=:black)
        figcontourdf2 = scatter(figcontourdf2,[r1],[r2],ms=10,legend=false,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,msw=0,color=:green2)
        display(figcontourdf2)
        savefig(figcontourdf2,filepath_save[1] * "figcontourdf2"   * ".pdf")

        # contour plot with univariate confidence level thresholds
        figcontourdf1 = contour(r1rangefull, r2rangefull, fullloglikelihood', levels=[-5.51,-3.32,-1.92,0],clims=(-10,0),xlabel=L"r_{1}", ylabel=L"r_{2}", fill=true,scalefont=fnt,colorbar_title=L"\hat{\ell}(\theta \ | \ y_{1:I}^{o})")
        figcontourdf1 = scatter(figcontourdf1,[r1mle],[r2mle],ms=10,color=:black)
        figcontourdf1 = scatter(figcontourdf1,[r1],[r2],ms=10,legend=false,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,msw=0,color=:green2)
        display(figcontourdf1)
        savefig(figcontourdf1,filepath_save[1] * "figcontourdf1"   * ".pdf")



        ### Prediction - confidence sets for the model solution, and confidence sets for data realisations

       
      
        predict_modelsolution=zeros(2,length(t_smooth),length(r1rangefull),length(r2rangefull))
        predict_realisations_min=1000*ones(2,length(t_smooth),length(r1rangefull),length(r2rangefull))
        predict_realisations_max=zeros(2,length(t_smooth),length(r1rangefull),length(r2rangefull))
        
        fullloglikelihood= zeros(length(r1rangefull),length(r2rangefull))
        for i=1:length(r1rangefull)
            for j=1:length(r2rangefull)
                    a = [r1rangefull[i],r2rangefull[j]]
                    fullloglikelihood[i,j] = error(data,a) - fopt;
                    if fullloglikelihood[i,j] > TH_full_95
                        # predict model solution
                        predict_modelsolution[:,:,i,j]=model(t_smooth,a)

                        if  fullloglikelihood[i,j] > TH_full_975
                            # predict data realisations
                            loop_data_dists=[Normal(mi,Sd) for mi in model(t_smooth,a)]; # normal distribution about mean
                            predict_realisations_min[:,:,i,j]= [quantile(data_dist, TH_full_alpha/2) for data_dist in loop_data_dists];                
                            predict_realisations_max[:,:,i,j]= [quantile(data_dist, 1-TH_full_alpha/2) for data_dist in loop_data_dists];                
                        end
                    end
            end
        end

        # combine the lower and upper
        max_modelsolution=zeros(2,length(t_smooth))
        min_modelsolution=1000*ones(2,length(t_smooth))
        for i in 1:length(r1rangefull)
            for j in 1:length(r2rangefull)
                if fullloglikelihood[i,j] > TH_full_95
                    for kk in 1:length(t_smooth)
                        max_modelsolution[1,kk]=max(predict_modelsolution[1,kk,i,j],max_modelsolution[1,kk])
                        min_modelsolution[1,kk]=min(predict_modelsolution[1,kk,i,j],min_modelsolution[1,kk])
                        max_modelsolution[2,kk]=max(predict_modelsolution[2,kk,i,j],max_modelsolution[2,kk])
                        min_modelsolution[2,kk]=min(predict_modelsolution[2,kk,i,j],min_modelsolution[2,kk])
                    end
                end
            end
        end

        # combine the confidence sets for realisations
        max_realisations=zeros(2,length(t_smooth))
        min_realisations=1000*ones(2,length(t_smooth))
        for i in 1:length(r1rangefull)
            for j in 1:length(r2rangefull)
                if fullloglikelihood[i,j] > TH_full_975
                    for kk in 1:length(t_smooth)
                        max_realisations[1,kk]=max(predict_realisations_max[1,kk,i,j],max_realisations[1,kk])
                        min_realisations[1,kk]=min(predict_realisations_min[1,kk,i,j],min_realisations[1,kk])
                        max_realisations[2,kk]=max(predict_realisations_max[2,kk,i,j],max_realisations[2,kk])
                        min_realisations[2,kk]=min(predict_realisations_min[2,kk,i,j],min_realisations[2,kk])
                    end
                end
            end
        end


        # plot the confidence sets for the mean
        ymle_smooth =  model(t_smooth,[r1mle,r2mle,C10,C20]); # MLE

        fullconfmodel=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1) # solid - c1
        fullconfmodel=plot!(t_smooth,ymle_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
        fullconfmodel=scatter!(t,data[1,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
        fullconfmodel=scatter!(t,data[2,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,msw=0) # scatter c2
        fullconfmodel=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_modelsolution[1,:],max_modelsolution[1,:].-ymle_smooth[1,:]),fillalpha=.2)
        fullconfmodel=plot!(t_smooth,ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_modelsolution[2,:],max_modelsolution[2,:].-ymle_smooth[2,:]),fillalpha=.2)

        display(fullconfmodel)
        savefig(fullconfmodel,filepath_save[1] * "Fig_fullconfmodel"   * ".pdf")

        # plot the difference  confidence sets for the mean and ymle

        ydiffmin=-5
        ydiffmax=5


        # union
        fullconfmodeldiff=plot(t_smooth,0 .*ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_modelsolution[1,:],max_modelsolution[1,:].-ymle_smooth[1,:]),ylims=(ydiffmin,ydiffmax),legend=false,fillalpha=.2,lw=0,linecolor=colour1)
        fullconfmodeldiff=plot!(t_smooth,0 .*ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_modelsolution[2,:],max_modelsolution[2,:].-ymle_smooth[2,:]),fillalpha=.2,xlab=L"t",ylab=L"C_{y,0.95} - y(\hat{\theta})",lw=0,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2)
        fullconfmodeldiff=plot!(t_smooth,0 .*ymle_smooth[2,:],lw=1,c=:black,linestyle=:dash)

        display(fullconfmodeldiff)
        savefig(fullconfmodeldiff,filepath_save[1] * "Fig_fullconfmodeldiff"   * ".pdf")

        # plot the confidence sets for the data realisations
        fullconfreal=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1) # solid - c1
        fullconfreal=plot!(t_smooth,ymle_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
        fullconfreal=scatter!(t,data[1,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
        fullconfreal=scatter!(t,data[2,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,msw=0) # scatter c2
        fullconfreal=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_realisations[1,:],max_realisations[1,:].-ymle_smooth[1,:]),fillalpha=.2)
        fullconfreal=plot!(t_smooth,ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_realisations[2,:],max_realisations[2,:].-ymle_smooth[2,:]),fillalpha=.2)
        
        display(fullconfreal)
        savefig(fullconfreal,filepath_save[1] * "Fig_fullconfreal"   * ".pdf")

    


#     #######################################################################################
#     #######################################################################################
#     #######################################################################################
     # PROFILE LIKELIHOOD APPROACH
     
     ## Profiling (using the MLE as the first guess at each point)
       
        nptss=40;

        #Profile r1

        r1min=lb[1]
        r1max=ub[1]
        r1range_lower=reverse(LinRange(r1min,r1mle,nptss))
        r1range_upper=LinRange(r1mle + (r1max-r1mle)/nptss,r1max,nptss)
        
        nr1range_lower=zeros(1,nptss)
        llr1_lower=zeros(nptss)
        nllr1_lower=zeros(nptss)
        predict_r1_lower=zeros(2,length(t_smooth),nptss)
        predict_r1_realisations_lower_lq=zeros(2,length(t_smooth),nptss)
        predict_r1_realisations_lower_uq=zeros(2,length(t_smooth),nptss)
        
        nr1range_upper=zeros(1,nptss)
        llr1_upper=zeros(nptss)
        nllr1_upper=zeros(nptss)
        predict_r1_upper=zeros(2,length(t_smooth),nptss)
        predict_r1_realisations_upper_lq=zeros(2,length(t_smooth),nptss)
        predict_r1_realisations_upper_uq=zeros(2,length(t_smooth),nptss)

        
        # start at mle and increase parameter (upper)
        for i in 1:nptss
            function fun1(aa)
                return error(data,[r1range_upper[i],aa[1],Sd])
            end
            lb1=[lb[2]];
            ub1=[ub[2]];
            local θG1=[r2mle]
            local (xo,fo)=optimise(fun1,θG1,lb1,ub1)
            nr1range_upper[:,i]=xo[:]
            llr1_upper[i]=fo[1]
            predict_r1_upper[:,:,i]=model(t_smooth,[r1range_upper[i],nr1range_upper[1,i]])

            loop_data_dists=[Normal(mi,Sd) for mi in model(t_smooth,[r1range_upper[i],nr1range_upper[1,i]])]; # normal distribution about mean
            predict_r1_realisations_upper_lq[:,:,i]= [quantile(data_dist, TH_prof_alpha/2) for data_dist in loop_data_dists];
            predict_r1_realisations_upper_uq[:,:,i]= [quantile(data_dist, 1-TH_prof_alpha/2) for data_dist in loop_data_dists];
            if fo > fmle
                println("new MLE r1 upper")
                global fmle=fo
                global r1mle=r1range_upper[i]
                global r2mle=nr1range_upper[1,i]
            end
        end
        
        # start at mle and decrease parameter (lower)
        for i in 1:nptss
            function fun1a(aa)
                return error(data,[r1range_lower[i],aa[1],Sd])
            end
            lb1=[lb[2]];
            ub1=[ub[2]];
            local θG1=[r2mle]
            local (xo,fo)=optimise(fun1a,θG1,lb1,ub1)
            nr1range_lower[:,i]=xo[:]
            llr1_lower[i]=fo[1]
            predict_r1_lower[:,:,i]=model(t_smooth,[r1range_lower[i],nr1range_lower[1,i]])

            loop_data_dists=[Normal(mi,Sd) for mi in model(t_smooth,[r1range_lower[i],nr1range_lower[1,i]])]; # normal distribution about mean
            predict_r1_realisations_lower_lq[:,:,i]= [quantile(data_dist, TH_prof_alpha/2) for data_dist in loop_data_dists];
            predict_r1_realisations_lower_uq[:,:,i]= [quantile(data_dist, 1-TH_prof_alpha/2) for data_dist in loop_data_dists];

            if fo > fmle
                println("new MLE r2 lower")
                global fmle = fo
                global r1mle=r1range_lower[i]
                global r2mle=nr1range_lower[1,i]
            end
        end
        
        # combine the lower and upper
        r1range = [reverse(r1range_lower); r1range_upper]
        nr1range = [reverse(nr1range_lower); nr1range_upper ]
        llr1 = [reverse(llr1_lower); llr1_upper] 
        nllr1=llr1.-maximum(llr1);
        
        max_r1=zeros(2,length(t_smooth))
        min_r1=1000*ones(2,length(t_smooth))
        for i in 1:(nptss)
            if (llr1_lower[i].-maximum(llr1)) >= TH_prof_95
                for j in 1:length(t_smooth)
                    max_r1[1,j]=max(predict_r1_lower[1,j,i],max_r1[1,j])
                    min_r1[1,j]=min(predict_r1_lower[1,j,i],min_r1[1,j])
                    max_r1[2,j]=max(predict_r1_lower[2,j,i],max_r1[2,j])
                    min_r1[2,j]=min(predict_r1_lower[2,j,i],min_r1[2,j])
                end
            end
        end
        for i in 1:(nptss)
            if (llr1_upper[i].-maximum(llr1)) >= TH_prof_95
                for j in 1:length(t_smooth)
                    max_r1[1,j]=max(predict_r1_upper[1,j,i],max_r1[1,j])
                    min_r1[1,j]=min(predict_r1_upper[1,j,i],min_r1[1,j]) 
                    max_r1[2,j]=max(predict_r1_upper[2,j,i],max_r1[2,j])
                    min_r1[2,j]=min(predict_r1_upper[2,j,i],min_r1[2,j])
                end
            end
        end
        # combine the confidence sets for realisations
        max_realisations_r1=zeros(2,length(t_smooth))
        min_realisations_r1=1000*ones(2,length(t_smooth))
        for i in 1:(nptss)
            if (llr1_lower[i].-maximum(llr1)) >= TH_prof_975
                for j in 1:length(t_smooth)
                    max_realisations_r1[1,j]=max(predict_r1_realisations_lower_uq[1,j,i],max_realisations_r1[1,j])
                    min_realisations_r1[1,j]=min(predict_r1_realisations_lower_lq[1,j,i],min_realisations_r1[1,j])
                    max_realisations_r1[2,j]=max(predict_r1_realisations_lower_uq[2,j,i],max_realisations_r1[2,j])
                    min_realisations_r1[2,j]=min(predict_r1_realisations_lower_lq[2,j,i],min_realisations_r1[2,j])
                end
            end
        end
        for i in 1:(nptss)
            if (llr1_upper[i].-maximum(llr1)) >= TH_prof_975
                for j in 1:length(t_smooth)
                    max_realisations_r1[1,j]=max(predict_r1_realisations_upper_uq[1,j,i],max_realisations_r1[1,j])
                    min_realisations_r1[1,j]=min(predict_r1_realisations_upper_lq[1,j,i],min_realisations_r1[1,j]) 
                    max_realisations_r1[2,j]=max(predict_r1_realisations_upper_uq[2,j,i],max_realisations_r1[2,j])
                    min_realisations_r1[2,j]=min(predict_r1_realisations_upper_lq[2,j,i],min_realisations_r1[2,j])
                end
            end
        end


        #Profile r2
        r2min=lb[2]
        r2max=ub[2]
        r2range_lower=reverse(LinRange(r2min,r2mle,nptss))
        r2range_upper=LinRange(r2mle + (r2max-r2mle)/nptss,r2max,nptss)
        
        nr2range_lower=zeros(1,nptss)
        llr2_lower=zeros(nptss)
        nllr2_lower=zeros(nptss)
        predict_r2_lower=zeros(2,length(t_smooth),nptss)
        predict_r2_realisations_lower_lq=zeros(2,length(t_smooth),nptss)
        predict_r2_realisations_lower_uq=zeros(2,length(t_smooth),nptss)
        
        nr2range_upper=zeros(1,nptss)
        llr2_upper=zeros(nptss)
        nllr2_upper=zeros(nptss)
        predict_r2_upper=zeros(2,length(t_smooth),nptss)
        predict_r2_realisations_upper_lq=zeros(2,length(t_smooth),nptss)
        predict_r2_realisations_upper_uq=zeros(2,length(t_smooth),nptss)
        
        # start at mle and increase parameter (upper)
        for i in 1:nptss
            function fun2(aa)
                return error(data,[aa[1],r2range_upper[i],Sd])
            end
            lb1=[lb[1]];
            ub1=[ub[1]];
            local θG1=[r1mle]
            local (xo,fo)=optimise(fun2,θG1,lb1,ub1)
            nr2range_upper[:,i]=xo[:]
            llr2_upper[i]=fo[1]
            predict_r2_upper[:,:,i]=model(t_smooth,[nr2range_upper[1,i],r2range_upper[i]])

            loop_data_dists=[Normal(mi,Sd) for mi in model(t_smooth,[nr2range_upper[1,i],r2range_upper[i]])]; # normal distribution about mean
            predict_r2_realisations_upper_lq[:,:,i]= [quantile(data_dist, TH_prof_alpha/2) for data_dist in loop_data_dists];
            predict_r2_realisations_upper_uq[:,:,i]= [quantile(data_dist, 1-TH_prof_alpha/2) for data_dist in loop_data_dists];

            if fo > fmle
                println("new MLE r2 upper")
                global fmle = fo
                global r1mle=nr2range_upper[1,i]
                global r2mle=r2range_upper[i]
            end
        end
        
        # start at mle and decrease parameter (lower)
        for i in 1:nptss
            function fun2a(aa)
                return error(data,[aa[1],r2range_lower[i],Sd])
            end
            lb1=[lb[1]];
            ub1=[ub[1]];
            local θG1=[r1mle]
            local (xo,fo)=optimise(fun2a,θG1,lb1,ub1)
            nr2range_lower[:,i]=xo[:]
            llr2_lower[i]=fo[1]
            predict_r2_lower[:,:,i]=model(t_smooth,[nr2range_lower[1,i],r2range_lower[i]])

            loop_data_dists=[Normal(mi,Sd) for mi in model(t_smooth,[nr2range_lower[1,i],r2range_lower[i]])]; # normal distribution about mean
            predict_r2_realisations_lower_lq[:,:,i]= [quantile(data_dist, TH_prof_alpha/2) for data_dist in loop_data_dists];
            predict_r2_realisations_lower_uq[:,:,i]= [quantile(data_dist, 1-TH_prof_alpha/2) for data_dist in loop_data_dists];
			

            if fo > fmle
                println("new MLE r2 lower")
                global fmle = fo
                global r1mle=nr2range_lower[1,i]
                global r2mle=r2range_lower[i]
            end
        end
        
        # combine the lower and upper
        r2range = [reverse(r2range_lower);r2range_upper]
        nr2range = [reverse(nr2range_lower); nr2range_upper ]
        llr2 = [reverse(llr2_lower); llr2_upper]     
        nllr2=llr2.-maximum(llr2);

        max_r2=zeros(2,length(t_smooth))
        min_r2=1000*ones(2,length(t_smooth))
        for i in 1:(nptss)
            if (llr2_lower[i].-maximum(llr2)) >= TH_prof_95
                for j in 1:length(t_smooth)
                    max_r2[1,j]=max(predict_r2_lower[1,j,i],max_r2[1,j])
                    min_r2[1,j]=min(predict_r2_lower[1,j,i],min_r2[1,j])
                    max_r2[2,j]=max(predict_r2_lower[2,j,i],max_r2[2,j])
                    min_r2[2,j]=min(predict_r2_lower[2,j,i],min_r2[2,j])
                end
            end
        end
        for i in 1:(nptss)
            if (llr2_upper[i].-maximum(llr2)) >= TH_prof_95
                for j in 1:length(t_smooth)
                    max_r2[1,j]=max(predict_r2_upper[1,j,i],max_r2[1,j])
                    min_r2[1,j]=min(predict_r2_upper[1,j,i],min_r2[1,j])
                    max_r2[2,j]=max(predict_r2_upper[2,j,i],max_r2[2,j])
                    min_r2[2,j]=min(predict_r2_upper[2,j,i],min_r2[2,j])
                end
            end
        end
        # combine the confidence sets for realisations
        max_realisations_r2=zeros(2,length(t_smooth))
        min_realisations_r2=1000*ones(2,length(t_smooth))
        for i in 1:(nptss)
            if (llr2_lower[i].-maximum(llr2)) >= TH_prof_975
                for j in 1:length(t_smooth)
                    max_realisations_r2[1,j]=max(predict_r2_realisations_lower_uq[1,j,i],max_realisations_r2[1,j])
                    min_realisations_r2[1,j]=min(predict_r2_realisations_lower_lq[1,j,i],min_realisations_r2[1,j])
                    max_realisations_r2[2,j]=max(predict_r2_realisations_lower_uq[2,j,i],max_realisations_r2[2,j])
                    min_realisations_r2[2,j]=min(predict_r2_realisations_lower_lq[2,j,i],min_realisations_r2[2,j])
                end
            end
        end
        for i in 1:(nptss)
            if (llr2_upper[i].-maximum(llr2)) >= TH_prof_975
                for j in 1:length(t_smooth)
                    max_realisations_r2[1,j]=max(predict_r2_realisations_upper_uq[1,j,i],max_realisations_r2[1,j])
                    min_realisations_r2[1,j]=min(predict_r2_realisations_upper_lq[1,j,i],min_realisations_r2[1,j]) 
                    max_realisations_r2[2,j]=max(predict_r2_realisations_upper_uq[2,j,i],max_realisations_r2[2,j])
                    min_realisations_r2[2,j]=min(predict_r2_realisations_upper_lq[2,j,i],min_realisations_r2[2,j])
                end
            end
        end
        

        ####################################
        # plot profiles
        combined_plot_linewidth = 2;

        # interpolate for smoother profile likelihoods
        interp_nptss= 1001;
        
        # r1
        interp_points_r1range =  LinRange(r1min,r1max,interp_nptss)
        interp_r1 = LinearInterpolation(r1range,nllr1)
        interp_nllr1 = interp_r1(interp_points_r1range)
        
        profile1=plot(interp_points_r1range,interp_nllr1,xlim=(0.8,1.2),ylim=(-4,0.1),yticks=[-3,-2,-1,0],xlab=L"r_{1}",ylab=L"\hat{\ell}_{p}",legend=false,lw=5,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=:deepskyblue3)
        profile1=hline!([-1.92],lw=2,linecolor=:black,linestyle=:dot)
        profile1=vline!([r1mle],lw=3,linecolor=:red)
        profile1=vline!([r1],lw=3,linecolor=:rosybrown,linestyle=:dash)

        # r2
        interp_points_r2range =  LinRange(r2min,r2max,interp_nptss)
        interp_r2 = LinearInterpolation(r2range,nllr2)
        interp_nllr2 = interp_r2(interp_points_r2range)

        profile2=plot(interp_points_r2range,interp_nllr2,xlim=(0.3,0.7),ylim=(-4,0.1),yticks=[-3,-2,-1,0],xlab=L"r_{2}",ylab=L"\hat{\ell}_{p}",legend=false,lw=5,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=:deepskyblue3)
        profile2=hline!([-1.92],lw=2,linecolor=:black,linestyle=:dot)
        profile2=vline!([r2mle],lw=3,linecolor=:red)
        profile1=vline!([r2],lw=3,linecolor=:rosybrown,linestyle=:dash)
        
        # Recompute best fit
        data0_smooth_MLE_recomputed = model(t_smooth,[r1mle,r2mle,C10,C20]); # generate data using known parameter values for plotting


        # plot model simulated at MLE and save
        p1_updated=plot(t_smooth,data0_smooth_MLE_recomputed[1,:],lw=3,linecolor=colour1) # solid - c1
        p1_updated=plot!(t_smooth,data0_smooth_MLE_recomputed[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
        p1_updated=scatter!(t,data[1,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,markerstrokewidth=0) # scatter c1
        p1_updated=scatter!(t,data[2,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,markerstrokewidth=0) # scatter c2


      
        # save figures        
        display(p1_updated)
        savefig(p1_updated,filepath_save[1] * "Figp1_updated" * ".pdf")

        display(profile1)
        savefig(profile1,filepath_save[1] * "Fig_profile1"   * ".pdf")
        
        display(profile2)
        savefig(profile2,filepath_save[1] * "Fig_profile2"   * ".pdf")
        

        # plot residuals
        t_residuals = t;
        data0_MLE_recomputed_for_residuals = model(t,[r1mle,r2mle,C10,C20]); # generate data using known parameter values for plotting
        data_residuals = data-data0_MLE_recomputed_for_residuals

        p_residuals=scatter(t,data_residuals[1,:],xlims=(xmin,xmax),ylims=(-30,30),legend=false,markersize = 15,markercolor=colour1,markerstrokewidth=0) # scatter c1
        p_residuals=scatter!(t,data_residuals[2,:],xlims=(xmin,xmax),ylims=(-30,30),legend=false,markersize =15,markercolor=colour2,xlab=L"t",ylab=L"e",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,markerstrokewidth=0) # scatter c2        
        p_residuals=hline!([0],lw=2,linecolor=:black,linestyle=:dot)

        display(p_residuals)
        savefig(p_residuals,filepath_save[1] * "Figp_residuals"   * ".pdf")

        # plot qq-plot
        p_residuals_qqplot = plot(qqplot(Normal,[data_residuals[1,:];data_residuals[2,:]],lw=2,linecolor=:black,linestyle=:dot,markersize = 8,markercolor=:black),legend=false,xlab=L"\mathrm{Normal}",ylab=L"\mathrm{Residuals}",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt)
        display(p_residuals_qqplot)
        savefig(p_residuals_qqplot,filepath_save[1] * "Figp_residuals_qqplot"   * ".pdf")

        
        #######################################################################################
        # compute the bounds of confidence interval
        
        function fun_interpCI(mle,interp_points_range,interp_nll,TH1)
            # find bounds of CI
            range_minus_mle = interp_points_range - mle*ones(length(interp_points_range),1)
            abs_range_minus_mle = broadcast(abs, range_minus_mle)
            findmin_mle = findmin(abs_range_minus_mle)
        
            # find closest value to CI threshold intercept
            value_minus_threshold = interp_nll - TH1*ones(length(interp_nll),1)
            abs_value_minus_threshold = broadcast(abs, value_minus_threshold)
            lb_CI_tmp = findmin(abs_value_minus_threshold[1:findmin_mle[2][1]])
            ub_CI_tmp = findmin(abs_value_minus_threshold[findmin_mle[2][1]:length(abs_value_minus_threshold)])
            lb_CI = interp_points_range[lb_CI_tmp[2][1]]
            ub_CI = interp_points_range[findmin_mle[2][1]-1 + ub_CI_tmp[2][1]]
        
            return lb_CI,ub_CI
        end
        
        # r1
        (lb_CI_r1,ub_CI_r1) = fun_interpCI(r1mle,interp_points_r1range,interp_nllr1,TH_prof_95)
        println(round(lb_CI_r1; digits = 4))
        println(round(ub_CI_r1; digits = 4))
        
        # r2
        (lb_CI_r2,ub_CI_r2) = fun_interpCI(r2mle,interp_points_r2range,interp_nllr2,TH_prof_95)
        println(round(lb_CI_r2; digits = 3))
        println(round(ub_CI_r2; digits = 3))
        
        
        # Export MLE and bounds to csv (one file for all data) -- -MLE ONLY 
        if @isdefined(df_MLEBoundsAll) == 0
            println("not defined")
            global df_MLEBoundsAll = DataFrame( r1mle=r1mle, lb_CI_r1=lb_CI_r1, ub_CI_r1=ub_CI_r1, r2mle=r2mle, lb_CI_r2=lb_CI_r2, ub_CI_r2=ub_CI_r2)
        else 
            println("DEFINED")
            global df_MLEBoundsAll_thisrow = DataFrame( r1mle=r1mle, lb_CI_r1=lb_CI_r1, ub_CI_r1=ub_CI_r1, r2mle=r2mle, lb_CI_r2=lb_CI_r2, ub_CI_r2=ub_CI_r2)
            append!(df_MLEBoundsAll,df_MLEBoundsAll_thisrow)
        end
        
        CSV.write(filepath_save[1] * "MLEBoundsALL.csv", df_MLEBoundsAll)

#########################################################################################
        # Parameter-wise profile predictions
       
        ymle_smooth =  model(t_smooth,[r1mle,r2mle,C10,C20]); # MLE

        # r1 - plot model simulated at MLE, scatter data, and prediction interval.
        confmodel1=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1) # solid - c1
        confmodel1=plot!(t_smooth,ymle_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
        confmodel1=scatter!(t,data[1,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
        confmodel1=scatter!(t,data[2,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,msw=0) # scatter c2 
        confmodel1=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_r1[1,:],max_r1[1,:].-ymle_smooth[1,:]),fillalpha=.2)
        confmodel1=plot!(t_smooth,ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_r1[2,:],max_r1[2,:].-ymle_smooth[2,:]),fillalpha=.2)
        
        display(confmodel1)
        savefig(confmodel1,filepath_save[1] * "Fig_confmodel1"   * ".pdf")

        # r2 - plot model simulated at MLE, scatter data, and prediction interval.
        confmodel2=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1) # solid - c1
        confmodel2=plot!(t_smooth,ymle_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
        confmodel2=scatter!(t,data[1,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
        confmodel2=scatter!(t,data[2,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,msw=0) # scatter c2
        confmodel2=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_r2[1,:],max_r2[1,:].-ymle_smooth[1,:]),fillalpha=.2)
        confmodel2=plot!(t_smooth,ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_r2[2,:],max_r2[2,:].-ymle_smooth[2,:]),fillalpha=.2)
        
        display(confmodel2)
        savefig(confmodel2,filepath_save[1] * "Fig_confmodel2"   * ".pdf")

        # overall - plot model simulated at MLE, scatter data, and prediction interval.
        max_overall=zeros(2,length(t_smooth))
        min_overall=1000*ones(2,length(t_smooth))
        for j in 1:length(t_smooth)
            max_overall[1,j]=max(max_r1[1,j],max_r2[1,j])
            max_overall[2,j]=max(max_r1[2,j],max_r2[2,j])
            min_overall[1,j]=min(min_r1[1,j],min_r2[1,j])
            min_overall[2,j]=min(min_r1[2,j],min_r2[2,j])
        end
        confmodelu=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1) # solid - c1
        confmodelu=plot!(t_smooth,ymle_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
        confmodelu=scatter!(t,data[1,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
        confmodelu=scatter!(t,data[2,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,msw=0) # scatter c2
        confmodelu=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_overall[1,:],max_overall[1,:].-ymle_smooth[1,:]),fillalpha=.2)
        confmodelu=plot!(t_smooth,ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_overall[2,:],max_overall[2,:].-ymle_smooth[2,:]),fillalpha=.2)

          display(confmodelu)
          savefig(confmodelu,filepath_save[1] * "Fig_confmodelu"   * ".pdf")

         


          ##############################
          # plot difference of confidence sets for models solutions ribbon to zero
        ydiffmin=-3.5
        ydiffmax=3.5

          # r1
          confmodeldiff1=plot(t_smooth,0 .* ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_r1[1,:],max_r1[1,:].-ymle_smooth[1,:]),ylims=(ydiffmin,ydiffmax),legend=false,fillalpha=.2,lw=0,linecolor=colour1)
          confmodeldiff1=plot!(t_smooth,0 .*ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_r1[2,:],max_r1[2,:].-ymle_smooth[2,:]),fillalpha=.2,xlab=L"t",ylab=L"C_{y,0.95}^{r_{1}} - y(\hat{\theta})",lw=0,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2)
          confmodeldiff1=plot!(t_smooth,0 .*ymle_smooth[2,:],lw=1,c=:black,linestyle=:dash)
        
        display(confmodeldiff1)
        savefig(confmodeldiff1,filepath_save[1] * "Fig_confmodeldiff1"   * ".pdf")

        # r2 
        confmodeldiff2=plot(t_smooth,0 .*ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_r2[1,:],max_r2[1,:].-ymle_smooth[1,:]),ylims=(ydiffmin,ydiffmax),legend=false,fillalpha=.2,lw=0,linecolor=colour1)
        confmodeldiff2=plot!(t_smooth,0 .*ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_r2[2,:],max_r2[2,:].-ymle_smooth[2,:]),fillalpha=.2,xlab=L"t",ylab=L"C_{y,0.95}^{r_{2}} - y(\hat{\theta})",lw=0,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2)
        confmodeldiff2=plot!(t_smooth,0 .*ymle_smooth[2,:],lw=1,c=:black,linestyle=:dash)
        
        display(confmodeldiff2)
        savefig(confmodeldiff2,filepath_save[1] * "Fig_confmodeldiff2"   * ".pdf")

        # union
        confmodeldiffu=plot(t_smooth,0 .*ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_overall[1,:],max_overall[1,:].-ymle_smooth[1,:]),ylims=(ydiffmin,ydiffmax),legend=false,fillalpha=.2,lw=0,linecolor=colour1)
        confmodeldiffu=plot!(t_smooth,0 .*ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_overall[2,:],max_overall[2,:].-ymle_smooth[2,:]),fillalpha=.2,xlab=L"t",ylab=L"C_{y,0.95} - y(\hat{\theta})",lw=0,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2)
        confmodeldiffu=plot!(t_smooth,0 .*ymle_smooth[2,:],lw=1,c=:black,linestyle=:dash)

          display(confmodeldiffu)
          savefig(confmodeldiffu,filepath_save[1] * "Fig_confmodeldiffu"   * ".pdf")



#########################################################################################
        # Confidence sets for model realisations


             # r1
             confmodelreal1=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1) # solid - c1
             confmodelreal1=plot!(t_smooth,ymle_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
             confmodelreal1=scatter!(t,data[1,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
             confmodelreal1=scatter!(t,data[2,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,msw=0) # scatter c2
             confmodelreal1=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_realisations_r1[1,:],max_realisations_r1[1,:].-ymle_smooth[1,:]),fillalpha=.2)
             confmodelreal1=plot!(t_smooth,ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_realisations_r1[2,:],max_realisations_r1[2,:].-ymle_smooth[2,:]),fillalpha=.2)
           
            display(confmodelreal1)
            savefig(confmodelreal1,filepath_save[1] * "Fig_confmodelreal1"   * ".pdf")

            # r2
            confmodelreal2=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1) # solid - c1
            confmodelreal2=plot!(t_smooth,ymle_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
            confmodelreal2=scatter!(t,data[1,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
            confmodelreal2=scatter!(t,data[2,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,msw=0) # scatter c2
            confmodelreal2=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_realisations_r2[1,:],max_realisations_r2[1,:].-ymle_smooth[1,:]),fillalpha=.2)
            confmodelreal2=plot!(t_smooth,ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_realisations_r2[2,:],max_realisations_r2[2,:].-ymle_smooth[2,:]),fillalpha=.2)
           
            display(confmodelreal2)
            savefig(confmodelreal2,filepath_save[1] * "Fig_confmodelreal2"   * ".pdf")


               # overall - plot model simulated at MLE, scatter data, and prediction interval (union of confidence sets for realisations)
            max_realisations_overall=zeros(2,length(t_smooth))
            min_realisations_overall=1000*ones(2,length(t_smooth))
            for j in 1:length(t_smooth)
                max_realisations_overall[1,j]=max(max_realisations_r1[1,j],max_realisations_r2[1,j])
                max_realisations_overall[2,j]=max(max_realisations_r1[2,j],max_realisations_r2[2,j])
                min_realisations_overall[1,j]=min(min_realisations_r1[1,j],min_realisations_r2[1,j])
                min_realisations_overall[2,j]=min(min_realisations_r1[2,j],min_realisations_r2[2,j])
            end

            confmodelrealu=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1) # solid - c1
            confmodelrealu=plot!(t_smooth,ymle_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
            confmodelrealu=scatter!(t,data[1,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
            confmodelrealu=scatter!(t,data[2,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,msw=0) # scatter c2
            confmodelrealu=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_realisations_overall[1,:],max_realisations_overall[1,:].-ymle_smooth[1,:]),fillalpha=.2)
            confmodelrealu=plot!(t_smooth,ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_realisations_overall[2,:],max_realisations_overall[2,:].-ymle_smooth[2,:]),fillalpha=.2)
           
            display(confmodelrealu)
            savefig(confmodelrealu,filepath_save[1] * "Fig_confmodelrealu"   * ".pdf")

            
#########################################################################################
        # Plot difference of confidence set for model realisations and the model solution at MLE 

            ynoisediffmin=-15
            ynoisediffmax=15

            confmodelrealdiff1=plot(t_smooth,0 .* ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_realisations_r1[1,:],max_realisations_r1[1,:].-ymle_smooth[1,:]),ylims=(ynoisediffmin,ynoisediffmax),legend=false,fillalpha=.2,lw=0,linecolor=colour1)
            confmodelrealdiff1=plot!(t_smooth,0 .* ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_realisations_r1[2,:],max_realisations_r1[2,:].-ymle_smooth[2,:]),fillalpha=.2,xlab=L"t",ylab=L"C_{y_{i},0.95}^{r_{1}} - y(\hat{\theta})",lw=0,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2)
            confmodelrealdiff1=plot!(t_smooth,0 .*ymle_smooth[2,:],lw=1,c=:black,linestyle=:dash)

            display(confmodelrealdiff1)
            savefig(confmodelrealdiff1,filepath_save[1] * "Fig_confmodelrealdiff1"   * ".pdf")

            # r2
            confmodelrealdiff2=plot(t_smooth,0 .* ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_realisations_r2[1,:],max_realisations_r2[1,:].-ymle_smooth[1,:]),ylims=(ynoisediffmin,ynoisediffmax),legend=false,fillalpha=.2,lw=0,linecolor=colour1)
            confmodelrealdiff2=plot!(t_smooth,0 .* ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_realisations_r2[2,:],max_realisations_r2[2,:].-ymle_smooth[2,:]),fillalpha=.2,xlab=L"t",ylab=L"C_{y_{i},0.95}^{r_{2}} - y(\hat{\theta})",lw=0,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2)
            confmodelrealdiff2=plot!(t_smooth,0 .*ymle_smooth[2,:],lw=1,c=:black,linestyle=:dash)

            display(confmodelrealdiff2)
            savefig(confmodelrealdiff2,filepath_save[1] * "Fig_confmodelrealdiff2"   * ".pdf")


             # union 
            confmodelrealdiffu=plot(t_smooth,0 .* ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_realisations_overall[1,:],max_realisations_overall[1,:].-ymle_smooth[1,:]),ylims=(ynoisediffmin,ynoisediffmax),legend=false,fillalpha=.2,lw=0,linecolor=colour1)
            confmodelrealdiffu=plot!(t_smooth,0 .* ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_realisations_overall[2,:],max_realisations_overall[2,:].-ymle_smooth[2,:]),fillalpha=.2,xlab=L"t",ylab=L"C_{y_{i},0.95} - y(\hat{\theta})",lw=0,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2)
            confmodelrealdiffu=plot!(t_smooth,0 .*ymle_smooth[2,:],lw=1,c=:black,linestyle=:dash)

            display(confmodelrealdiffu)
            savefig(confmodelrealdiffu,filepath_save[1] * "Fig_confmodelrealdiffu"   * ".pdf")

             
    #########################################################################################
    #########################################################################################
    #########################################################################################
      # COMPARISON OF (FULL LIKELIHOOD RESULTS) VS (UNION OF PROFILE-WISE CONFIDENCE SETS)
       # CONFIDENCE SETS FOR model solutions

      comp_fullprofile_modelsolution=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1) # solid - c1
      comp_fullprofile_modelsolution=plot!(t_smooth,ymle_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
      comp_fullprofile_modelsolution=scatter!(t,data[1,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
      comp_fullprofile_modelsolution=scatter!(t,data[2,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,msw=0) # scatter c2
      # full
      
      comp_fullprofile_modelsolution=plot!(t_smooth,min_modelsolution[1,:],linecolor=:black,linestyle=:dash,lw=2)
      comp_fullprofile_modelsolution=plot!(t_smooth,max_modelsolution[1,:],linecolor=:black,linestyle=:dash,lw=2)
      comp_fullprofile_modelsolution=plot!(t_smooth,min_modelsolution[2,:],linecolor=:black,linestyle=:dash,lw=2)
      comp_fullprofile_modelsolution=plot!(t_smooth,max_modelsolution[2,:],linecolor=:black,linestyle=:dash,lw=2)
      # profile
      comp_fullprofile_modelsolution=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_overall[1,:],max_overall[1,:].-ymle_smooth[1,:]),fillalpha=.3)
      comp_fullprofile_modelsolution=plot!(t_smooth,ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_overall[2,:],max_overall[2,:].-ymle_smooth[2,:]),fillalpha=.3)
    
      display(comp_fullprofile_modelsolution)
      savefig(comp_fullprofile_modelsolution,filepath_save[1] * "Fig_comp_fullprofile_modelsolution"   * ".pdf")
      
      



        # CONFIDENCE SETS FOR DATA realisations


        comp_fullprofile_realisations=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1) # solid - c1
        comp_fullprofile_realisations=plot!(t_smooth,ymle_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
        comp_fullprofile_realisations=scatter!(t,data[1,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
        comp_fullprofile_realisations=scatter!(t,data[2,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,msw=0) # scatter c2
        # full
        # comp_fullprofile_realisations=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_realisations[1,:],max_realisations[1,:].-ymle_smooth[1,:]),fillalpha=.2)
        # comp_fullprofile_realisations=plot!(t_smooth,ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_realisations[2,:],max_realisations[2,:].-ymle_smooth[2,:]),fillalpha=.2)
        comp_fullprofile_realisations=plot!(t_smooth,min_realisations[1,:],linecolor=:black,linestyle=:dash,lw=2)
        comp_fullprofile_realisations=plot!(t_smooth,max_realisations[1,:],linecolor=:black,linestyle=:dash,lw=2)
        comp_fullprofile_realisations=plot!(t_smooth,min_realisations[2,:],linecolor=:black,linestyle=:dash,lw=2)
        comp_fullprofile_realisations=plot!(t_smooth,max_realisations[2,:],linecolor=:black,linestyle=:dash,lw=2)
        # profile
        comp_fullprofile_realisations=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_realisations_overall[1,:],max_realisations_overall[1,:].-ymle_smooth[1,:]),fillalpha=.2)
        comp_fullprofile_realisations=plot!(t_smooth,ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_realisations_overall[2,:],max_realisations_overall[2,:].-ymle_smooth[2,:]),fillalpha=.2)
     
        display(comp_fullprofile_realisations)
        savefig(comp_fullprofile_realisations,filepath_save[1] * "Fig_comp_fullprofile_realisations"   * ".pdf")


             
    ##############################
      # COMPARISON OF (FULL LIKELIHOOD RESULTS) VS (UNION OF PROFILE-WISE CONFIDENCE SETS) 
      # profile likelihood v maximum over  full likelihood grid

        # interpolate for smoother profile likelihoods
        interp_nptss= 1001;
        
        # r1
        comp_fullprofile_s1=plot(interp_points_r1range,interp_nllr1,xlim=(0.8,1.2),ylim=(-4,0.1),yticks=[-3,-2,-1,0],xlab=L"r_{1}",ylab=L"\hat{\ell}_{p}",legend=false,lw=5,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=:deepskyblue3)
        comp_fullprofile_s1=hline!([-1.92],lw=2,linecolor=:black,linestyle=:dot)
        comp_fullprofile_s1=vline!([r1mle],lw=3,linecolor=:red)


        full_maxloglikelihood_r1 = -1000 .*ones(length(r1rangefull))
        for i in 1:length(r1rangefull)
            for j in 1:length(r2rangefull)
                full_maxloglikelihood_r1[i] = max(fullloglikelihood[i,j],full_maxloglikelihood_r1[i])
            end
        end

        comp_fullprofile_s1=plot!(r1rangefull,full_maxloglikelihood_r1,lw=5, linecolor=:black,linestyle=:dash)

        display(comp_fullprofile_s1)
        savefig(comp_fullprofile_s1,filepath_save[1] * "Fig_comp_fullprofile_s1"   * ".pdf")

        # r2
        
        comp_fullprofile_s2=plot(interp_points_r2range,interp_nllr2,xlim=(0.3,0.7),ylim=(-4,0.1),yticks=[-3,-2,-1,0],xlab=L"r_{2}",ylab=L"\hat{\ell}_{p}",legend=false,lw=5,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=:deepskyblue3)
        comp_fullprofile_s2=hline!([-1.92],lw=2,linecolor=:black,linestyle=:dot)
        comp_fullprofile_s2=vline!([r2mle],lw=3,linecolor=:red)

        full_maxloglikelihood_r2= -1000 .*ones(length(r2rangefull))
        for i in 1:length(r1rangefull)
            for j in 1:length(r2rangefull)
                full_maxloglikelihood_r2[j] = max(fullloglikelihood[i,j],full_maxloglikelihood_r2[j])
            end
        end

        comp_fullprofile_s2=plot!(r2rangefull,full_maxloglikelihood_r2,lw=5, linecolor=:black,linestyle=:dash)


        display(comp_fullprofile_s2)
        savefig(comp_fullprofile_s2,filepath_save[1] * "Fig_comp_fullprofile_s2"   * ".pdf")
        
