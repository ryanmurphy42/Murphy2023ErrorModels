# Murphy et al. (2023) 
# Implementing measurement error models with mechanistic mathematical models
# in a likelihood-based framework for estimation, identifiability analysis, 
# and prediction in the life sciences

# Fig5_ODELinear_LogNormalNoise_LogNormalFit

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
global cur_colors = palette(:default) # plot options
isdir(pwd() * "\\Fig5_ODELinear_LogNormalNoise_LogNormalFit\\") || mkdir(pwd() * "\\Fig5_ODELinear_LogNormalNoise_LogNormalFit") # make folder to save figures if doesnt already exist
filepath_save = [pwd() * "\\Fig5_ODELinear_LogNormalNoise_LogNormalFit\\"] # location to save figures "\\Fig2\\"] # location to save figures

#######################################################################################
## Mathematical model

a = zeros(3);
r1=1.0;
r2=0.5;
C10=100.0;
C20=10.0;
t_max = 5.0;
t=LinRange(0.0,t_max, 31); # synthetic data measurement times
t_smooth =  LinRange(0.0,t_max, 101); # for plotting known values and best-fits

Sd = 0.4;

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

Random.seed!(1) # set random seed

data0=0.0*zeros(2,length(t)); # generate vector to data from known parameters
data=0.0*zeros(2,length(t)); # generate vector to data from known parameters + noise

data0=model(t,[r1,r2]); # generate data using known parameter values to add noise
data0_smooth = model(t_smooth,[r1,r2]); # generate data using known parameter values for plotting

data = [ mi*rand(LogNormal(0,Sd)) for mi in model(t,[r1,r2])]; # generate the noisy data

global df_data = DataFrame(c1 = data[1,:], c2=data[2,:])
CSV.write(filepath_save[1] * "Data.csv", df_data)
 

#######################################################################################
### Plot data - scatter plot

colour1=:green2
colour2=:magenta

p0=plot(t_smooth,data0_smooth[1,:],lw=3,linecolor=colour1) # solid - c1
p0=plot!(t_smooth,data0_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
p0=scatter!(t,data[1,:],xlims=(0,5),ylims=(0,125),legend=false,markersize = 8,markercolor=colour1,markerstrokewidth=0) # scatter c1
p0=scatter!(t,data[2,:],xlims=(0,5.5),ylims=(0,125),legend=false,markersize = 8,markercolor=colour2,markerstrokewidth=0) # scatter c2

display(p0)
savefig(p0,filepath_save[1] * "Fig0" * ".pdf")

#######################################################################################
### Parameter identifiability analysis - MLE, profiles

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

#
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
    global fmle=fopt
    global r1mle=xopt[1]
    global r2mle=xopt[2]
    global Sdmle=xopt[3]


    data0_smooth_MLE = model(t_smooth,[r1mle,r2mle]); # generate data using known parameter values for plotting

    # plot model simulated at MLE and data
    p1=plot(t_smooth,data0_smooth_MLE[1,:],lw=3,linecolor=colour1) # solid - c1
    p1=plot!(t_smooth,data0_smooth_MLE[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
    p1=scatter!(t,data[1,:],xlims=(0,5),ylims=(0,125),legend=false,markersize = 8,markercolor=colour1,markerstrokewidth=0) # scatter c1
    p1=scatter!(t,data[2,:],xlims=(0,5.5),ylims=(0,125),legend=false,markersize = 8,markercolor=colour2,markerstrokewidth=0) # scatter c2
    
    display(p1)
    savefig(p1,filepath_save[1] * "Fig1" * ".pdf")



    #######################################################################################
        # Profiling (using the MLE as the first guess at each point)
       
        nptss=20;

        #Profile r1

        r1min=lb[1]
        r1max=ub[1]
        r1range_lower=reverse(LinRange(r1min,r1mle,nptss))
        r1range_upper=LinRange(r1mle + (r1max-r1mle)/nptss,r1max,nptss)
        

        nr1range_lower=zeros(2,nptss)
        llr1_lower=zeros(nptss)
        nllr1_lower=zeros(nptss)
        predict_r1_lower=zeros(2,length(t_smooth),nptss)
        predict_r1_realisations_lower_lq=zeros(2,length(t_smooth),nptss)
        predict_r1_realisations_lower_uq=zeros(2,length(t_smooth),nptss)

        nr1range_upper=zeros(2,nptss)
        llr1_upper=zeros(nptss)
        nllr1_upper=zeros(nptss)
        predict_r1_upper=zeros(2,length(t_smooth),nptss)
        predict_r1_realisations_upper_lq=zeros(2,length(t_smooth),nptss)
        predict_r1_realisations_upper_uq=zeros(2,length(t_smooth),nptss)
        
        # start at mle and increase parameter (upper)
        for i in 1:nptss
            function fun1(aa)
                return error(data,[r1range_upper[i],aa[1],aa[2]])
            end
            lb1=[lb[2],lb[3]];
            ub1=[ub[2],ub[3]];
            local θG1=[r2mle,Sdmle]
            local (xo,fo)=optimise(fun1,θG1,lb1,ub1)
            nr1range_upper[:,i]=xo[:]
            llr1_upper[i]=fo[1]

            modelmean = model(t_smooth,[r1range_upper[i],nr1range_upper[1,i]]);
            predict_r1_upper[:,:,i]=modelmean;

            loop_data_dists=[LogNormal(0,nr1range_upper[2,i]) for mi in modelmean]; 
            predict_r1_realisations_upper_lq[:,:,i]= [quantile(data_dist, THalpha/2) for data_dist in loop_data_dists].*modelmean;
            predict_r1_realisations_upper_uq[:,:,i]= [quantile(data_dist, 1-THalpha/2) for data_dist in loop_data_dists].*modelmean;

            if fo > fmle
                println("new MLE r1 upper")
                global fmle=fo
                global r1mle=r1range_upper[i]
                global r2mle=nr1range_upper[1,i]
                global Sdmle=nr1range_upper[2,i]
            end
        end
        
        # start at mle and decrease parameter (lower)
        for i in 1:nptss
            function fun1a(aa)
                return error(data,[r1range_lower[i],aa[1],aa[2]])
            end
            lb1=[lb[2],lb[3]];
            ub1=[ub[2],ub[3]];
            local θG1=[r2mle, Sdmle]
            local (xo,fo)=optimise(fun1a,θG1,lb1,ub1)
            nr1range_lower[:,i]=xo[:]
            llr1_lower[i]=fo[1]
            
            modelmean = model(t_smooth,[r1range_lower[i],nr1range_lower[1,i]]);
            predict_r1_lower[:,:,i]=modelmean;

            loop_data_dists=[LogNormal(0,nr1range_lower[2,i]) for mi in modelmean]; 
            predict_r1_realisations_lower_lq[:,:,i]= [quantile(data_dist, THalpha/2) for data_dist in loop_data_dists].*modelmean;
            predict_r1_realisations_lower_uq[:,:,i]= [quantile(data_dist, 1-THalpha/2) for data_dist in loop_data_dists].*modelmean;

            if fo > fmle
                println("new MLE r2 lower")
                global fmle = fo
                global r1mle=r1range_lower[i]
                global r2mle=nr1range_lower[1,i]
                global Sdmle=nr1range_lower[2,i]
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
            if (llr1_lower[i].-maximum(llr1)) >= TH
                for j in 1:length(t_smooth)
                    max_r1[1,j]=max(predict_r1_lower[1,j,i],max_r1[1,j])
                    min_r1[1,j]=min(predict_r1_lower[1,j,i],min_r1[1,j])
                    max_r1[2,j]=max(predict_r1_lower[2,j,i],max_r1[2,j])
                    min_r1[2,j]=min(predict_r1_lower[2,j,i],min_r1[2,j])
                end
            end
        end
        for i in 1:(nptss)
            if (llr1_upper[i].-maximum(llr1)) >= TH
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
            if (llr1_lower[i].-maximum(llr1)) >= TH_realisations
                for j in 1:length(t_smooth)
                    max_realisations_r1[1,j]=max(predict_r1_realisations_lower_uq[1,j,i],max_realisations_r1[1,j])
                    min_realisations_r1[1,j]=min(predict_r1_realisations_lower_lq[1,j,i],min_realisations_r1[1,j])
                    max_realisations_r1[2,j]=max(predict_r1_realisations_lower_uq[2,j,i],max_realisations_r1[2,j])
                    min_realisations_r1[2,j]=min(predict_r1_realisations_lower_lq[2,j,i],min_realisations_r1[2,j])
                end
            end
        end
        for i in 1:(nptss)
            if (llr1_upper[i].-maximum(llr1)) >= TH_realisations
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
        
        nr2range_lower=zeros(2,nptss)
        llr2_lower=zeros(nptss)
        nllr2_lower=zeros(nptss)
        predict_r2_lower=zeros(2,length(t_smooth),nptss)
        predict_r2_realisations_lower_lq=zeros(2,length(t_smooth),nptss)
        predict_r2_realisations_lower_uq=zeros(2,length(t_smooth),nptss)
        
        nr2range_upper=zeros(2,nptss)
        llr2_upper=zeros(nptss)
        nllr2_upper=zeros(nptss)
        predict_r2_upper=zeros(2,length(t_smooth),nptss)
        predict_r2_realisations_upper_lq=zeros(2,length(t_smooth),nptss)
        predict_r2_realisations_upper_uq=zeros(2,length(t_smooth),nptss)
        
        # start at mle and increase parameter (upper)
        for i in 1:nptss
            function fun2(aa)
                return error(data,[aa[1],r2range_upper[i],aa[2]])
            end
            lb1=[lb[1],lb[3]];
            ub1=[ub[1],ub[3]];
            local θG1=[r1mle, Sdmle]
            local (xo,fo)=optimise(fun2,θG1,lb1,ub1)
            nr2range_upper[:,i]=xo[:]
            llr2_upper[i]=fo[1]

			modelmean = model(t_smooth,[nr2range_upper[1,i],r2range_upper[i]]);
            predict_r2_upper[:,:,i]=modelmean;

            loop_data_dists=[LogNormal(0,nr2range_upper[2,i]) for mi in modelmean]; 
            predict_r2_realisations_upper_lq[:,:,i]= [quantile(data_dist, THalpha/2) for data_dist in loop_data_dists].*modelmean;
            predict_r2_realisations_upper_uq[:,:,i]= [quantile(data_dist, 1-THalpha/2) for data_dist in loop_data_dists].*modelmean;

            if fo > fmle
                println("new MLE r2 upper")
                global fmle = fo
                global r1mle=nr2range_upper[1,i]
                global r2mle=r2range_upper[i]
                global Sdmle=nr2range_upper[2,i]
            end
        end
        
        # start at mle and decrease parameter (lower)
        for i in 1:nptss
            function fun2a(aa)
                return error(data,[aa[1],r2range_lower[i],aa[2]])
            end
            lb1=[lb[1],lb[3]];
            ub1=[ub[1],ub[3]];
            local θG1=[r1mle, Sdmle]
            local (xo,fo)=optimise(fun2a,θG1,lb1,ub1)
            nr2range_lower[:,i]=xo[:]
            llr2_lower[i]=fo[1]

            modelmean = model(t_smooth,[nr2range_lower[1,i],r2range_lower[i]]);
            predict_r2_lower[:,:,i]=modelmean;

            loop_data_dists=[LogNormal(0,nr2range_lower[2,i]) for mi in modelmean]; 
            predict_r2_realisations_lower_lq[:,:,i]= [quantile(data_dist, THalpha/2) for data_dist in loop_data_dists].*modelmean;
            predict_r2_realisations_lower_uq[:,:,i]= [quantile(data_dist, 1-THalpha/2) for data_dist in loop_data_dists].*modelmean;

            if fo > fmle
                println("new MLE r2 lower")
                global fmle = fo
                global r1mle=nr2range_lower[1,i]
                global r2mle=r2range_lower[i]
                global Sdmle=nr2range_lower[2,i]
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
            if (llr2_lower[i].-maximum(llr2)) >= TH
                for j in 1:length(t_smooth)
                    max_r2[1,j]=max(predict_r2_lower[1,j,i],max_r2[1,j])
                    min_r2[1,j]=min(predict_r2_lower[1,j,i],min_r2[1,j])
                    max_r2[2,j]=max(predict_r2_lower[2,j,i],max_r2[2,j])
                    min_r2[2,j]=min(predict_r2_lower[2,j,i],min_r2[2,j])
                end
            end
        end
        for i in 1:(nptss)
            if (llr2_upper[i].-maximum(llr2)) >= TH
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
            if (llr2_lower[i].-maximum(llr2)) >= TH_realisations
                for j in 1:length(t_smooth)
                    max_realisations_r2[1,j]=max(predict_r2_realisations_lower_uq[1,j,i],max_realisations_r2[1,j])
                    min_realisations_r2[1,j]=min(predict_r2_realisations_lower_lq[1,j,i],min_realisations_r2[1,j])
                    max_realisations_r2[2,j]=max(predict_r2_realisations_lower_uq[2,j,i],max_realisations_r2[2,j])
                    min_realisations_r2[2,j]=min(predict_r2_realisations_lower_lq[2,j,i],min_realisations_r2[2,j])
                end
            end
        end
        for i in 1:(nptss)
            if (llr2_upper[i].-maximum(llr2)) >= TH_realisations
                for j in 1:length(t_smooth)
                    max_realisations_r2[1,j]=max(predict_r2_realisations_upper_uq[1,j,i],max_realisations_r2[1,j])
                    min_realisations_r2[1,j]=min(predict_r2_realisations_upper_lq[1,j,i],min_realisations_r2[1,j]) 
                    max_realisations_r2[2,j]=max(predict_r2_realisations_upper_uq[2,j,i],max_realisations_r2[2,j])
                    min_realisations_r2[2,j]=min(predict_r2_realisations_upper_lq[2,j,i],min_realisations_r2[2,j])
                end
            end
        end


        #Profile Sd
        Sdmin=lb[3]
        Sdmax=ub[3]
        Sdrange_lower=reverse(LinRange(Sdmin,Sdmle,nptss))
        Sdrange_upper=LinRange(Sdmle + (Sdmax-Sdmle)/nptss,Sdmax,nptss)
        
        nSdrange_lower=zeros(2,nptss)
        llSd_lower=zeros(nptss)
        nllSd_lower=zeros(nptss)
        predict_Sd_lower=zeros(2,length(t_smooth),nptss)
        predict_Sd_realisations_lower_lq=zeros(2,length(t_smooth),nptss)
        predict_Sd_realisations_lower_uq=zeros(2,length(t_smooth),nptss)

        nSdrange_upper=zeros(2,nptss)
        llSd_upper=zeros(nptss)
        nllSd_upper=zeros(nptss)
        predict_Sd_upper=zeros(2,length(t_smooth),nptss)
        predict_Sd_realisations_upper_lq=zeros(2,length(t_smooth),nptss)
        predict_Sd_realisations_upper_uq=zeros(2,length(t_smooth),nptss)

        
        # start at mle and increase parameter (upper)
        for i in 1:nptss
            function fun3(aa)
                return error(data,[aa[1],aa[2],Sdrange_upper[i]])
            end
            lb1=[lb[1],lb[2]];
            ub1=[ub[1],ub[2]];
            local θG1=[r1mle,r2mle]    
            local (xo,fo)=optimise(fun3,θG1,lb1,ub1)
            nSdrange_upper[:,i]=xo[:]
            llSd_upper[i]=fo[1]

            modelmean = model(t_smooth,[nSdrange_upper[1,i],nSdrange_upper[2,i]])
            predict_Sd_upper[:,:,i]=modelmean;

            loop_data_dists=[LogNormal(0,Sdrange_upper[i]) for mi in modelmean]; 
            predict_Sd_realisations_upper_lq[:,:,i]= [quantile(data_dist, THalpha/2) for data_dist in loop_data_dists].*modelmean;
            predict_Sd_realisations_upper_uq[:,:,i]= [quantile(data_dist, 1-THalpha/2) for data_dist in loop_data_dists].*modelmean;

            if fo > fmle
                println("new MLE Sd upper")
                global fmle = fo
                global r1mle=nSdrange_upper[1,i]
                global r2mle=nSdrange_upper[2,i]
                global Sdmle=Sdrange_upper[i]
            end
        end
        
        # start at mle and decrease parameter (lower)
        for i in 1:nptss
            function fun3a(aa)
                return error(data,[aa[1],aa[2],Sdrange_lower[i]])
            end
            lb1=[lb[1],lb[2]];
            ub1=[ub[1],ub[2]];
            local θG1=[r1mle,r2mle]      
            local (xo,fo)=optimise(fun3a,θG1,lb1,ub1)
            nSdrange_lower[:,i]=xo[:]
            llSd_lower[i]=fo[1]

            modelmean = model(t_smooth,[nSdrange_lower[1,i],nSdrange_lower[2,i]])
            predict_Sd_lower[:,:,i]=modelmean;

            loop_data_dists=[LogNormal(0,Sdrange_lower[i]) for mi in modelmean]; 
            predict_Sd_realisations_lower_lq[:,:,i]= [quantile(data_dist, THalpha/2) for data_dist in loop_data_dists].*modelmean;
            predict_Sd_realisations_lower_uq[:,:,i]= [quantile(data_dist, 1-THalpha/2) for data_dist in loop_data_dists].*modelmean;

            if fo > fmle
                println("new MLE Sd lower")
                global fmle = fo
                global r1mle=nSdrange_lower[1,i]
                global r2mle=nSdrange_lower[2,i]
                global Sdmle=Sdrange_lower[i]
            end
        end
        
        # combine the lower and upper
        Sdrange = [reverse(Sdrange_lower);Sdrange_upper]
        nSdrange = [reverse(nSdrange_lower); nSdrange_upper ]
        llSd = [reverse(llSd_lower); llSd_upper] 
        
        nllSd=llSd.-maximum(llSd)


        # combine the lower and upper
        Sdrange = [reverse(Sdrange_lower); Sdrange_upper]
        nSdrange = [reverse(nSdrange_lower); nSdrange_upper ]
        llSd = [reverse(llSd_lower); llSd_upper] 
        nllSd=llSd.-maximum(llSd);

        max_Sd=zeros(2,length(t_smooth))
        min_Sd=1000*ones(2,length(t_smooth))
        for i in 1:(nptss)
            if (llSd_lower[i].-maximum(llSd)) >= TH
                for j in 1:length(t_smooth)
                    max_Sd[1,j]=max(predict_Sd_lower[1,j,i],max_Sd[1,j])
                    min_Sd[1,j]=min(predict_Sd_lower[1,j,i],min_Sd[1,j])
                    max_Sd[2,j]=max(predict_Sd_lower[2,j,i],max_Sd[2,j])
                    min_Sd[2,j]=min(predict_Sd_lower[2,j,i],min_Sd[2,j])
                end
            end
        end
        for i in 1:(nptss)
            if (llSd_upper[i].-maximum(llSd)) >= TH
                for j in 1:length(t_smooth)
                    max_Sd[1,j]=max(predict_Sd_upper[1,j,i],max_Sd[1,j])
                    min_Sd[1,j]=min(predict_Sd_upper[1,j,i],min_Sd[1,j]) 
                    max_Sd[2,j]=max(predict_Sd_upper[2,j,i],max_Sd[2,j])
                    min_Sd[2,j]=min(predict_Sd_upper[2,j,i],min_Sd[2,j])
                end
            end
        end

        # combine the confidence sets for realisations
        max_realisations_Sd=zeros(2,length(t_smooth))
        min_realisations_Sd=1000*ones(2,length(t_smooth))
        for i in 1:(nptss)
            if (llSd_lower[i].-maximum(llSd)) >= TH_realisations
                for j in 1:length(t_smooth)
                    max_realisations_Sd[1,j]=max(predict_Sd_realisations_lower_uq[1,j,i],max_realisations_Sd[1,j])
                    min_realisations_Sd[1,j]=min(predict_Sd_realisations_lower_lq[1,j,i],min_realisations_Sd[1,j])
                    max_realisations_Sd[2,j]=max(predict_Sd_realisations_lower_uq[2,j,i],max_realisations_Sd[2,j])
                    min_realisations_Sd[2,j]=min(predict_Sd_realisations_lower_lq[2,j,i],min_realisations_Sd[2,j])
                end
            end
        end
        for i in 1:(nptss)
            if (llSd_upper[i].-maximum(llSd)) >= TH_realisations
                for j in 1:length(t_smooth)
                    max_realisations_Sd[1,j]=max(predict_Sd_realisations_upper_uq[1,j,i],max_realisations_Sd[1,j])
                    min_realisations_Sd[1,j]=min(predict_Sd_realisations_upper_lq[1,j,i],min_realisations_Sd[1,j]) 
                    max_realisations_Sd[2,j]=max(predict_Sd_realisations_upper_uq[2,j,i],max_realisations_Sd[2,j])
                    min_realisations_Sd[2,j]=min(predict_Sd_realisations_upper_lq[2,j,i],min_realisations_Sd[2,j])
                end
            end
        end


        ##############################################################################################################
        # Plot Profile Likelihoods

        combined_plot_linewidth = 2;

        # interpolate for smoother profile likelihoods
        interp_nptss= 1001;
        
        # r1
        interp_points_r1range =  LinRange(r1min,r1max,interp_nptss)
        interp_r1 = LinearInterpolation(r1range,nllr1)
        interp_nllr1 = interp_r1(interp_points_r1range)
        
        profile1=plot(interp_points_r1range,interp_nllr1,xlim=(r1min,r1max),ylim=(-4,0.1),yticks=[-3,-2,-1,0],xlab=L"r_{1}",ylab=L"\hat{\ell}_{p}",legend=false,lw=5,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=:deepskyblue3)
        profile1=hline!([-1.92],lw=2,linecolor=:black,linestyle=:dot)
        profile1=vline!([r1mle],lw=3,linecolor=:red)
        profile1=vline!([r1],lw=3,linecolor=:rosybrown,linestyle=:dash)

        # r2
        interp_points_r2range =  LinRange(r2min,r2max,interp_nptss)
        interp_r2 = LinearInterpolation(r2range,nllr2)
        interp_nllr2 = interp_r2(interp_points_r2range)
        
        profile2=plot(interp_points_r2range,interp_nllr2,xlim=(r2min,r2max),ylim=(-4,0.1),yticks=[-3,-2,-1,0],xlab=L"r_{2}",ylab=L"\hat{\ell}_{p}",legend=false,lw=5,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=:deepskyblue3)
        profile2=hline!([-1.92],lw=2,linecolor=:black,linestyle=:dot)
        profile2=vline!([r2mle],lw=3,linecolor=:red)
        profile2=vline!([r2],lw=3,linecolor=:rosybrown,linestyle=:dash)

        
        # Sd
        interp_points_Sdrange =  LinRange(Sdmin,Sdmax,interp_nptss)
        interp_Sd = LinearInterpolation(Sdrange,nllSd)
        interp_nllSd = interp_Sd(interp_points_Sdrange)
        
        profile3=plot(interp_points_Sdrange,interp_nllSd,xlim=(Sdmin,Sdmax),ylim=(-4,0.1),yticks=[-3,-2,-1,0],xlab=L"\sigma_{L}",ylab=L"\hat{\ell}_{p}",legend=false,lw=5,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=:deepskyblue3)
        profile3=hline!([-1.92],lw=2,linecolor=:black,linestyle=:dot)
        profile3=vline!([Sdmle],lw=3,linecolor=:red)
        profile3=vline!([Sd],lw=3,linecolor=:rosybrown,linestyle=:dash)


        # Recompute best fit
        data0_smooth_MLE_recomputed = model(t_smooth,[r1mle,r2mle]); # generate data using known parameter values for plotting


        # plot model simulated at MLE and save
        p1_updated=plot(t_smooth,data0_smooth_MLE_recomputed[1,:],lw=3,linecolor=colour1) # solid - c1
        p1_updated=plot!(t_smooth,data0_smooth_MLE_recomputed[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
        p1_updated=scatter!(t,data[1,:],xlims=(0,5),ylims=(0,125),legend=false,markersize = 8,markercolor=colour1,markerstrokewidth=0) # scatter c1
        p1_updated=scatter!(t,data[2,:],xlims=(0,5.5),ylims=(0,125),legend=false,markersize = 8,markercolor=colour2,markerstrokewidth=0) # scatter c2
        p1_updated=plot!(t_smooth,data0_smooth[1,:],lw=2,linecolor=:black,linestyle=:dash) # dash - c1
        p1_updated=plot!(t_smooth,data0_smooth[2,:],lw=2,linecolor=:black,linestyle=:dash) # dash - c2
        


      
        # save figures        
        display(p1_updated)
        savefig(p1_updated,filepath_save[1] * "Figp1_updated" * ".pdf")

        display(profile1)
        savefig(profile1,filepath_save[1] * "Fig_profile1"   * ".pdf")
        
        display(profile2)
        savefig(profile2,filepath_save[1] * "Fig_profile2"   * ".pdf")
        
        display(profile3)
        savefig(profile3,filepath_save[1] * "Fig_profile3"   * ".pdf")


          # plot residuals
          t_residuals = t;
          data0_MLE_recomputed_for_residuals = model(t,[r1mle,r2mle]); # generate data using known parameter values for plotting
          data_residuals = data0_MLE_recomputed_for_residuals-data
          maximum(abs.(data_residuals))
          data_residuals_multiplicative = data./data0_MLE_recomputed_for_residuals
          maximum(data_residuals_multiplicative)
  
          p_residuals=scatter(t,data_residuals[1,:],xlims=(0,5),ylims=(-65,65),legend=false,markersize = 8,markercolor=colour1,markerstrokewidth=0) # scatter c1
          p_residuals=scatter!(t,data_residuals[2,:],xlims=(-0.1,5.5),ylims=(-65,65),legend=false,markersize = 8,markercolor=colour2,xlab=L"t",ylab=L"\hat{e}_{i}",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,markerstrokewidth=0) # scatter c2        
          p_residuals=hline!([0],lw=2,linecolor=:black,linestyle=:dot)
  
          display(p_residuals)
          savefig(p_residuals,filepath_save[1] * "Figp_residuals"   * ".pdf")
  
          # plot qq-plot
          p_residuals_qqplot = plot(qqplot(Normal,[data_residuals[1,:];data_residuals[2,:]],lw=2,linecolor=:black,linestyle=:dot,markersize = 8,markercolor=:black),legend=false,xlab=L"\mathrm{Normal}",ylab=L"\mathrm{Residuals}",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt)
          display(p_residuals_qqplot)
          savefig(p_residuals_qqplot,filepath_save[1] * "Figp_residuals_qqplot"   * ".pdf")

          # plot qq-plot lognormal
          p_residuals_qqplot_lognormal = plot(qqplot(LogNormal,[data_residuals_multiplicative[1,:];data_residuals_multiplicative[2,:]],lw=2,linecolor=:black,linestyle=:dot,markersize = 8,markercolor=:black),legend=false,xlab=L"\mathrm{LogNormal}",ylab=L"\hat{e}_{i}",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt)
          display(p_residuals_qqplot_lognormal)
          savefig(p_residuals_qqplot_lognormal,filepath_save[1] * "Figp_residuals_qqplot_lognormal"   * ".pdf")
        
        #######################################################################################
        # compute the bounds of confidence interval
        
        function fun_interpCI(mle,interp_points_range,interp_nll,TH)
            # find bounds of CI
            range_minus_mle = interp_points_range - mle*ones(length(interp_points_range),1)
            abs_range_minus_mle = broadcast(abs, range_minus_mle)
            findmin_mle = findmin(abs_range_minus_mle)
        
            # find closest value to CI threshold intercept
            value_minus_threshold = interp_nll - TH*ones(length(interp_nll),1)
            abs_value_minus_threshold = broadcast(abs, value_minus_threshold)
            lb_CI_tmp = findmin(abs_value_minus_threshold[1:findmin_mle[2][1]])
            ub_CI_tmp = findmin(abs_value_minus_threshold[findmin_mle[2][1]:length(abs_value_minus_threshold)])
            lb_CI = interp_points_range[lb_CI_tmp[2][1]]
            ub_CI = interp_points_range[findmin_mle[2][1]-1 + ub_CI_tmp[2][1]]
        
            return lb_CI,ub_CI
        end
        
        # r1
        (lb_CI_r1,ub_CI_r1) = fun_interpCI(r1mle,interp_points_r1range,interp_nllr1,TH)
        println(round(lb_CI_r1; digits = 4))
        println(round(ub_CI_r1; digits = 4))
        
        # r2
        (lb_CI_r2,ub_CI_r2) = fun_interpCI(r2mle,interp_points_r2range,interp_nllr2,TH)
        println(round(lb_CI_r2; digits = 3))
        println(round(ub_CI_r2; digits = 3))
        
        # Sd
        (lb_CI_Sd,ub_CI_Sd) = fun_interpCI(Sdmle,interp_points_Sdrange,interp_nllSd,TH)
        println(round(lb_CI_Sd; digits = 3))
        println(round(ub_CI_Sd; digits = 3))
        
        # Export MLE and bounds to csv (one file for all data) -- -MLE ONLY 
        if @isdefined(df_MLEBoundsAll) == 0
            println("not defined")
            global df_MLEBoundsAll = DataFrame( r1mle=r1mle, lb_CI_r1=lb_CI_r1, ub_CI_r1=ub_CI_r1, r2mle=r2mle, lb_CI_r2=lb_CI_r2, ub_CI_r2=ub_CI_r2, Sdmle=Sdmle, lb_CI_Sd=lb_CI_Sd, ub_CI_Sd=ub_CI_Sd)
        else 
            println("DEFINED")
            global df_MLEBoundsAll_thisrow = DataFrame( r1mle=r1mle, lb_CI_r1=lb_CI_r1, ub_CI_r1=ub_CI_r1, r2mle=r2mle, lb_CI_r2=lb_CI_r2, ub_CI_r2=ub_CI_r2, Sdmle=Sdmle, lb_CI_Sd=lb_CI_Sd, ub_CI_Sd=ub_CI_Sd)
            append!(df_MLEBoundsAll,df_MLEBoundsAll_thisrow)
        end
        
        CSV.write(filepath_save[1] * "MLEBoundsALL.csv", df_MLEBoundsAll)


        #########################################################################################
        # Confidence sets for model solutions

        ymin=-50
        ymax=250
       
        ymle_smooth =  model(t_smooth,[r1mle,r2mle]); # MLE

        # r1 - plot model simulated at MLE, scatter data, and confidence set
        confmodel1=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1) # solid - c1
        confmodel1=plot!(t_smooth,ymle_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
        confmodel1=scatter!(t,data[1,:],xlims=(0,5),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,markerstrokewidth=0) # scatter c1
        confmodel1=scatter!(t,data[2,:],xlims=(0,5.5),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,markerstrokewidth=0) # scatter c2
        confmodel1=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_r1[1,:],max_r1[1,:].-ymle_smooth[1,:]),fillalpha=.2)
        confmodel1=plot!(t_smooth,ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_r1[2,:],max_r1[2,:].-ymle_smooth[2,:]),fillalpha=.2)
        confmodel1=hline!([0],lw=2,linecolor=:black,linestyle=:dash)
        display(confmodel1)
        savefig(confmodel1,filepath_save[1] * "Fig_confmodel1"   * ".pdf")

        # r2 - plot model simulated at MLE, scatter data, and confidence set
        confmodel2=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1) # solid - c1
        confmodel2=plot!(t_smooth,ymle_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
        confmodel2=scatter!(t,data[1,:],xlims=(0,5),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,markerstrokewidth=0) # scatter c1
        confmodel2=scatter!(t,data[2,:],xlims=(0,5.5),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,markerstrokewidth=0) # scatter c2
        confmodel2=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_r2[1,:],max_r2[1,:].-ymle_smooth[1,:]),fillalpha=.2)
        confmodel2=plot!(t_smooth,ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_r2[2,:],max_r2[2,:].-ymle_smooth[2,:]),fillalpha=.2)
        confmodel2=hline!([0],lw=2,linecolor=:black,linestyle=:dash)
        display(confmodel2)
        savefig(confmodel2,filepath_save[1] * "Fig_confmodel2"   * ".pdf")


        # Sd - plot model simulated at MLE, scatter data, and confidence set
        confmodel3=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1) # solid - c1
        confmodel3=plot!(t_smooth,ymle_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
        confmodel3=scatter!(t,data[1,:],xlims=(0,5),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,markerstrokewidth=0) # scatter c1
        confmodel3=scatter!(t,data[2,:],xlims=(0,5.5),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,markerstrokewidth=0) # scatter c2
        confmodel3=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_Sd[1,:],max_Sd[1,:].-ymle_smooth[1,:]),fillalpha=.2)
        confmodel3=plot!(t_smooth,ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_Sd[2,:],max_Sd[2,:].-ymle_smooth[2,:]),fillalpha=.2)
        confmodel3=hline!([0],lw=2,linecolor=:black,linestyle=:dash)
        display(confmodel3)
        savefig(confmodel3,filepath_save[1] * "Fig_confmodel3"   * ".pdf")


        # union - plot model simulated at MLE, scatter data, and confidence set
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
          confmodelu=scatter!(t,data[1,:],xlims=(0,5),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,markerstrokewidth=0) # scatter c1
          confmodelu=scatter!(t,data[2,:],xlims=(0,5.5),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,markerstrokewidth=0) # scatter c2
          confmodelu=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_overall[1,:],max_overall[1,:].-ymle_smooth[1,:]),fillalpha=.2)
          confmodelu=plot!(t_smooth,ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_overall[2,:],max_overall[2,:].-ymle_smooth[2,:]),fillalpha=.2)
          confmodelu=hline!([0],lw=2,linecolor=:black,linestyle=:dash)
      
          display(confmodelu)
        savefig(confmodelu,filepath_save[1] * "Fig_confmodelu"   * ".pdf")


  
#########################################################################################
        # Plot difference of confidence set for model solutions and the model solution at MLE 

        xmin=0
        xmax=5.5

        ydiffmin=-7.5
        ydiffmax=7.5

        # r1
        confmodeldiff1=plot(t_smooth,0 .* ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_r1[1,:],max_r1[1,:].-ymle_smooth[1,:]),ylims=(ydiffmin,ydiffmax),legend=false,fillalpha=.2,lw=0,linecolor=colour1)
        confmodeldiff1=plot!(t_smooth,0 .* ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_r1[2,:],max_r1[2,:].-ymle_smooth[2,:]),fillalpha=.2,xlab=L"t",ylab=L"C_{y,0.95}^{r_{1}} - y(\hat{\theta})",lw=0,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2)
        confmodeldiff1=plot!(t_smooth,0 .*ymle_smooth[2,:],lw=1,c=:black,linestyle=:dash,xlims=(xmin,xmax),yticks=([-5,0,5]))
        
        display(confmodeldiff1)
        savefig(confmodeldiff1,filepath_save[1] * "Fig_confmodeldiff1"   * ".pdf")

        # r2 
        confmodeldiff2=plot(t_smooth,0 .* ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_r2[1,:],max_r2[1,:].-ymle_smooth[1,:]),ylims=(ydiffmin,ydiffmax),legend=false,fillalpha=.2,lw=0,linecolor=colour1)
        confmodeldiff2=plot!(t_smooth,0 .* ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_r2[2,:],max_r2[2,:].-ymle_smooth[2,:]),fillalpha=.2,xlab=L"t",ylab=L"C_{y,0.95}^{r_{2}} - y(\hat{\theta})",lw=0,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2)
        confmodeldiff2=plot!(t_smooth,0 .*ymle_smooth[2,:],lw=1,c=:black,linestyle=:dash,xlims=(xmin,xmax),yticks=([-5,0,5]))
        
        display(confmodeldiff2)
        savefig(confmodeldiff2,filepath_save[1] * "Fig_confmodeldiff2"   * ".pdf")

        # Sigma
        confmodeldiff3=plot(t_smooth,0 .* ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_Sd[1,:],max_Sd[1,:].-ymle_smooth[1,:]),ylims=(ydiffmin,ydiffmax),legend=false,fillalpha=.2,lw=0,linecolor=colour1)
        confmodeldiff3=plot!(t_smooth,0 .* ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_Sd[2,:],max_Sd[2,:].-ymle_smooth[2,:]),fillalpha=.2,xlab=L"t",ylab=L"C_{y,0.95}^{\sigma_{L}} - y(\hat{\theta})",lw=0,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2)
        confmodeldiff3=plot!(t_smooth,0 .*ymle_smooth[2,:],lw=1,c=:black,linestyle=:dash,xlims=(xmin,xmax),yticks=([-5,0,5]))
        
        display(confmodeldiff3)
        savefig(confmodeldiff3,filepath_save[1] * "Fig_confmodeldiff3"   * ".pdf")
   

        # union 
       
        confmodeldiffu=plot(t_smooth,0 .* ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_overall[1,:],max_overall[1,:].-ymle_smooth[1,:]),ylims=(ydiffmin,ydiffmax),legend=false,fillalpha=.2,lw=0,linecolor=colour1)
        confmodeldiffu=plot!(t_smooth,0 .* ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_overall[2,:],max_overall[2,:].-ymle_smooth[2,:]),fillalpha=.2,xlab=L"t",ylab=L"C_{y,0.95} - y(\hat{\theta})",lw=0,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2)
        confmodeldiffu=plot!(t_smooth,0 .*ymle_smooth[2,:],lw=1,c=:black,linestyle=:dash,xlims=(xmin,xmax),yticks=([-5,0,5]))

        display(confmodeldiffu)
        savefig(confmodeldiffu,filepath_save[1] * "Fig_confmodeldiffu"   * ".pdf")




#########################################################################################
        # Confidence sets for model realisations
        

            ymin=-50
            ymax=250

             # r1
             confreal1=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1) # solid - c1
             confreal1=plot!(t_smooth,ymle_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
             confreal1=scatter!(t,data[1,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
             confreal1=scatter!(t,data[2,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,msw=0) # scatter c2
             confreal1=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_realisations_r1[1,:],max_realisations_r1[1,:].-ymle_smooth[1,:]),fillalpha=.2)
             confreal1=plot!(t_smooth,ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_realisations_r1[2,:],max_realisations_r1[2,:].-ymle_smooth[2,:]),fillalpha=.2)
             confreal1=hline!([0],lw=2,linecolor=:black,linestyle=:dash)

             display(confreal1)
             savefig(confreal1,filepath_save[1] * "Fig_confreal1"   * ".pdf")
 
             # r2
             confreal2=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1) # solid - c1
             confreal2=plot!(t_smooth,ymle_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
             confreal2=scatter!(t,data[1,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
             confreal2=scatter!(t,data[2,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,msw=0) # scatter c2
             confreal2=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_realisations_r2[1,:],max_realisations_r2[1,:].-ymle_smooth[1,:]),fillalpha=.2)
             confreal2=plot!(t_smooth,ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_realisations_r2[2,:],max_realisations_r2[2,:].-ymle_smooth[2,:]),fillalpha=.2)
             confreal2=hline!([0],lw=2,linecolor=:black,linestyle=:dash)

             display(confreal2)
             savefig(confreal2,filepath_save[1] * "Fig_confreal2"   * ".pdf")
 
            #Sd
            confreal3=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1) # solid - c1
            confreal3=plot!(t_smooth,ymle_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
            confreal3=scatter!(t,data[1,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
            confreal3=scatter!(t,data[2,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,msw=0) # scatter c2
            confreal3=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_realisations_Sd[1,:],max_realisations_Sd[1,:].-ymle_smooth[1,:]),fillalpha=.2)
            confreal3=plot!(t_smooth,ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_realisations_Sd[2,:],max_realisations_Sd[2,:].-ymle_smooth[2,:]),fillalpha=.2)
            confreal3=hline!([0],lw=2,linecolor=:black,linestyle=:dash)

            display(confreal3)
            savefig(confreal3,filepath_save[1] * "Fig_confreal3"   * ".pdf")
 
 
            # union 
            max_realisations_overall=zeros(2,length(t_smooth))
            min_realisations_overall=1000*ones(2,length(t_smooth))
            for j in 1:length(t_smooth)
                 max_realisations_overall[1,j]=max(max_realisations_r1[1,j],max_realisations_r2[1,j],max_realisations_Sd[1,j])
                 max_realisations_overall[2,j]=max(max_realisations_r1[2,j],max_realisations_r2[2,j],max_realisations_Sd[2,j])
                 min_realisations_overall[1,j]=min(min_realisations_r1[1,j],min_realisations_r2[1,j],min_realisations_Sd[1,j])
                 min_realisations_overall[2,j]=min(min_realisations_r1[2,j],min_realisations_r2[2,j],min_realisations_Sd[2,j])
            end
 
            conufrealu=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1) # solid - c1
            conufrealu=plot!(t_smooth,ymle_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
            conufrealu=scatter!(t,data[1,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
            conufrealu=scatter!(t,data[2,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,msw=0) # scatter c2
            conufrealu=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_realisations_overall[1,:],max_realisations_overall[1,:].-ymle_smooth[1,:]),fillalpha=.2)
            conufrealu=plot!(t_smooth,ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_realisations_overall[2,:],max_realisations_overall[2,:].-ymle_smooth[2,:]),fillalpha=.2)
            confrealu=hline!([0],lw=2,linecolor=:black,linestyle=:dash)
            
            display(conufrealu)
            savefig(conufrealu,filepath_save[1] * "Fig_conufrealu"   * ".pdf")
 
 
 
 
 
             