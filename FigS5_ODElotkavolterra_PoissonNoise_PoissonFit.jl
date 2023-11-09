# Murphy et al. (2023) 
# Implementing measurement error models with mechanistic mathematical models
# in a likelihood-based framework for estimation, identifiability analysis, 
# and prediction in the life sciences

# FigS5_ODElotkavolterra_PoissonNoise_PoissonFit

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
isdir(pwd() * "\\FigS5_ODElotkavolterra_PoissonNoise_PoissonFit\\") || mkdir(pwd() * "\\FigS5_ODElotkavolterra_PoissonNoise_PoissonFit") # make folder to save figures if doesnt already exist
filepath_save = [pwd() * "\\FigS5_ODElotkavolterra_PoissonNoise_PoissonFit\\"] # location to save figures "\\Fig2\\"] # location to save figures


#######################################################################################
## Mathematical model

a = zeros(4);
V1=0.1;
K1=0.0025;
V2=0.0025;
K2=0.3;
C10=30.0;
C20=10.0;
t_max = 100.0;
t=[25*ones(3); 40*ones(3); 50*ones(3); 60*ones(3); 75*ones(3)]; # synthetic data measurement times
t_smooth =  LinRange(0.0,t_max*1.1, 101); # for plotting known values and best-fits

function DE!(du,u,p,t)
    V1,K1,V2,K2 = p
    du[1] = V1*u[1] - K1*u[1]*u[2];
    du[2] = V2*u[1]*u[2] - K2*u[2];
end

#######################################################################################
## Solving the mathematical model

function odesolver(t,V1,K1,V2,K2,C0)
    p=[V1,K1,V2,K2]
    tspan=(0.0,maximum(t));
    prob=ODEProblem(DE!,C0,tspan,p);
    sol=solve(prob,saveat=t);
    return sol[1:2,:];
end

function model(t,a)
    y=zeros(length(t))
    y=odesolver(t,a[1],a[2],a[3],a[4],[C10,C20])
    return y
end
    
#######################################################################################
## Generate Synthetic data

    Random.seed!(1)

    data0=0.0*zeros(2,length(t)); # generate vector to data from known parameters
    data=0.0*zeros(2,length(t)); # generate vector to data from known parameters + noise

    data0=model(t,[V1,K1,V2,K2,C10,C20]); # generate data using known parameter values to add noise
    data0[data0 .< 0] .= zero(eltype(data0))
    data0_smooth = model(t_smooth,[V1,K1,V2,K2,C10,C20]); # generate data using known parameter values for plotting

    data_dists=[Poisson(mi) for mi in data0] # generate noisy data using Poisson
    data=[rand(data_dist) for data_dist in data_dists]; # generate the noisy data

    global df_data = DataFrame(c1 = data[1,:], c2=data[2,:])
    CSV.write(filepath_save[1] * "Data.csv", df_data)

#######################################################################################
### Plot data - scatter plot

ymin=-100
ymax=410;

colour1=:green2
colour2=:magenta

p0=plot(t_smooth,data0_smooth[1,:],lw=3,linecolor=colour1) # solid - c1
p0=plot!(t_smooth,data0_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
p0=scatter!(t,data[1,:],xlims=(0,t_max*1.1),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
p0=scatter!(t,data[2,:],xlims=(0,t_max*1.1),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,msw=0) # scatter c2
p0=hline!([0],lw=2,linecolor=:black,linestyle=:dash)

display(p0)
savefig(p0,filepath_save[1] * "Fig0" * ".pdf")

#######################################################################################
### Parameter identifiability analysis - MLE, profiles

    # initial guesses for MLE search
    VV1 = V1*1.01;
    KK1 = K1*1.01;
    VV2 = V2*1.01;
    KK2 = K2*1.01;
    
    TH=-1.921; #95% confidence interval threshold (for confidence interval for model parameters, and confidence set for model solutions)
    TH_realisations = -2.51 # 97.5 confidence interval threshold (for model realisations)
    THalpha = 0.025; # 97.5% data realisations interval threshold

    function error(data,a)
        y=zeros(length(t))
        y=model(t,a);
        e=0;
        y[y .< 0] .= zero(eltype(y))
        data_dists=[Poisson(mi) for mi in y]; # Poisson noise model
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

    θG = [VV1,KK1,VV2,KK2] # first guess

    # lower and upper bounds for parameter estimation
    V1_lb = V1*0.5 
    V1_ub = V1*1.5 
    K1_lb = K1*0.3 
    K1_ub = K1*1.5 
    V2_lb = V2*0.5 
    V2_ub = V2*1.5 
    K2_lb = K2*0.5 
    K2_ub = K2*2.0 

    lb=[V1_lb,K1_lb,V2_lb,K2_lb];
    ub=[V1_ub,K1_ub,V2_ub,K2_ub];

    # MLE optimisation
    (xopt,fopt)  = optimise(fun,θG,lb,ub)

    # storing MLE 
    global fmle=fopt
    global V1mle=xopt[1]
    global K1mle=xopt[2]
    global V2mle=xopt[3]
    global K2mle=xopt[4]


    data0_smooth_MLE = model(t_smooth,[V1mle,K1mle,V2mle,K2mle,C10,C20]); # generate data using known parameter values for plotting


    # plot model simulated at MLE and save
    p1=plot(t_smooth,data0_smooth_MLE[1,:],lw=3,linecolor=colour1) # solid - c1
    p1=plot!(t_smooth,data0_smooth_MLE[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
    p1=scatter!(t,data[1,:],xlims=(0,t_max*1.1),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
    p1=scatter!(t,data[2,:],xlims=(0,t_max*1.1),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,msw=0) # scatter c2
    p1=hline!([0],lw=2,linecolor=:black,linestyle=:dash)

    display(p1)
    savefig(p1,filepath_save[1] * "Fig1" * ".pdf")





        #######################################################################################
        # Profiling (using the MLE as the first guess at each point)

        # PARAMETERS (P1, P2, P3, P4) = (V1, K1, V2, K2)
        global P1mle = V1mle
        global P2mle = K1mle
        global P3mle = V2mle
        global P4mle = K2mle
       
        nptss=20;

        #Profile V1

        P1min=lb[1]
        P1max=ub[1]
        P1range_lower=reverse(LinRange(P1min,P1mle,nptss))
        P1range_upper=LinRange(P1mle + (P1max-P1mle)/nptss,P1max,nptss)
        
        P1nrange_lower=zeros(3,nptss)
        llP1_lower=zeros(nptss)
        nllP1_lower=zeros(nptss)
        predict_P1_lower=zeros(2,length(t_smooth),nptss)
        predict_P1_realisations_lower_lq=zeros(2,length(t_smooth),nptss)
        predict_P1_realisations_lower_uq=zeros(2,length(t_smooth),nptss)
        
        
        P1nrange_upper=zeros(3,nptss)
        llP1_upper=zeros(nptss)
        nllP1_upper=zeros(nptss)
        predict_P1_upper=zeros(2,length(t_smooth),nptss)
        predict_P1_realisations_upper_lq=zeros(2,length(t_smooth),nptss)
        predict_P1_realisations_upper_uq=zeros(2,length(t_smooth),nptss)
        
        # start at mle and increase parameter (upper)
        for i in 1:nptss
            function fun1(aa)
                return error(data,[P1range_upper[i],aa[1],aa[2],aa[3]])
            end
            lb1=[lb[2],lb[3],lb[4]];
            ub1=[ub[2],ub[3],ub[4]];
            local θG1=[P2mle, P3mle, P4mle]
            local (xo,fo)=optimise(fun1,θG1,lb1,ub1)
            P1nrange_upper[:,i]=xo[:]
            llP1_upper[i]=fo[1]

            modelmean = model(t_smooth,[P1range_upper[i],P1nrange_upper[1,i],P1nrange_upper[2,i],P1nrange_upper[3,i]])
            predict_P1_upper[:,:,i]= modelmean

            loop_data_dists=[Poisson(mi) for mi in modelmean]; 
            predict_P1_realisations_upper_lq[:,:,i]= [quantile(data_dist, THalpha/2) for data_dist in loop_data_dists];
            predict_P1_realisations_upper_uq[:,:,i]= [quantile(data_dist, 1-THalpha/2) for data_dist in loop_data_dists];
            
            if fo > fmle
                println("new MLE P1 upper")
                global fmle=fo
                global P1mle=P1range_upper[i]
                global P2mle=P1nrange_upper[1,i]
                global P3mle=P1nrange_upper[2,i]
                global P4mle=P1nrange_upper[3,i]
            end
        end
        
        # start at mle and decrease parameter (lower)
        for i in 1:nptss
            function fun1a(aa)
                return error(data,[P1range_lower[i],aa[1],aa[2],aa[3]])
            end
            lb1=[lb[2],lb[3],lb[4]];
            ub1=[ub[2],ub[3],ub[4]];
            local θG1=[P2mle, P3mle, P4mle]
            local (xo,fo)=optimise(fun1a,θG1,lb1,ub1)
            P1nrange_lower[:,i]=xo[:]
            llP1_lower[i]=fo[1]
            
            modelmean = model(t_smooth,[P1range_lower[i],P1nrange_lower[1,i],P1nrange_lower[2,i],P1nrange_lower[3,i]])
            predict_P1_lower[:,:,i]= modelmean

            loop_data_dists=[Poisson(mi) for mi in modelmean]; # normal distribution about mean
            predict_P1_realisations_lower_lq[:,:,i]= [quantile(data_dist, THalpha/2) for data_dist in loop_data_dists];
            predict_P1_realisations_lower_uq[:,:,i]= [quantile(data_dist, 1-THalpha/2) for data_dist in loop_data_dists];


            if fo > fmle
                println("new MLE P1 lower")
                global fmle = fo
                global P1mle=P1range_lower[i]
                global P2mle=P1nrange_lower[1,i]
                global P3mle=P1nrange_lower[2,i]
                global P4mle=P1nrange_lower[3,i]
            end
        end
        
        # combine the lower and upper
        P1range = [reverse(P1range_lower); P1range_upper]
        P1nrange = [reverse(P1nrange_lower); P1nrange_upper ]
        llP1 = [reverse(llP1_lower); llP1_upper] 
        nllP1=llP1.-maximum(llP1);
        # min and max for confidence set for model solutions
        max_P1=zeros(2,length(t_smooth))
        min_P1=100000*ones(2,length(t_smooth))
        for i in 1:(nptss)
            if (llP1_lower[i].-maximum(llP1)) >= TH
                for j in 1:length(t_smooth)
                    max_P1[1,j]=max(predict_P1_lower[1,j,i],max_P1[1,j])
                    min_P1[1,j]=min(predict_P1_lower[1,j,i],min_P1[1,j])
                    max_P1[2,j]=max(predict_P1_lower[2,j,i],max_P1[2,j])
                    min_P1[2,j]=min(predict_P1_lower[2,j,i],min_P1[2,j])
                end
            end
        end
        for i in 1:(nptss)
            if (llP1_upper[i].-maximum(llP1)) >= TH
                for j in 1:length(t_smooth)
                    max_P1[1,j]=max(predict_P1_upper[1,j,i],max_P1[1,j])
                    min_P1[1,j]=min(predict_P1_upper[1,j,i],min_P1[1,j]) 
                    max_P1[2,j]=max(predict_P1_upper[2,j,i],max_P1[2,j])
                    min_P1[2,j]=min(predict_P1_upper[2,j,i],min_P1[2,j])
                end
            end
        end
         # combine the confidence sets for realisations
         max_realisations_P1=zeros(2,length(t_smooth))
         min_realisations_P1=1000*ones(2,length(t_smooth))
         for i in 1:(nptss)
             if (llP1_lower[i].-maximum(llP1)) >= TH_realisations
                 for j in 1:length(t_smooth)
                     max_realisations_P1[1,j]=max(predict_P1_realisations_lower_uq[1,j,i],max_realisations_P1[1,j])
                     min_realisations_P1[1,j]=min(predict_P1_realisations_lower_lq[1,j,i],min_realisations_P1[1,j])
                     max_realisations_P1[2,j]=max(predict_P1_realisations_lower_uq[2,j,i],max_realisations_P1[2,j])
                     min_realisations_P1[2,j]=min(predict_P1_realisations_lower_lq[2,j,i],min_realisations_P1[2,j])
                 end
             end
         end
         for i in 1:(nptss)
             if (llP1_upper[i].-maximum(llP1)) >= TH_realisations
                 for j in 1:length(t_smooth)
                     max_realisations_P1[1,j]=max(predict_P1_realisations_upper_uq[1,j,i],max_realisations_P1[1,j])
                     min_realisations_P1[1,j]=min(predict_P1_realisations_upper_lq[1,j,i],min_realisations_P1[1,j]) 
                     max_realisations_P1[2,j]=max(predict_P1_realisations_upper_uq[2,j,i],max_realisations_P1[2,j])
                     min_realisations_P1[2,j]=min(predict_P1_realisations_upper_lq[2,j,i],min_realisations_P1[2,j])
                 end
             end
         end

        #Profile K1
        P2min=lb[2]
        P2max=ub[2]
        P2range_lower=reverse(LinRange(P2min,P2mle,nptss))
        P2range_upper=LinRange(P2mle + (P2max-P2mle)/nptss,P2max,nptss)
        
        P2nrange_lower=zeros(3,nptss)
        llP2_lower=zeros(nptss)
        nllP2_lower=zeros(nptss)
        predict_P2_lower=zeros(2,length(t_smooth),nptss)
        predict_P2_realisations_lower_lq=zeros(2,length(t_smooth),nptss)
        predict_P2_realisations_lower_uq=zeros(2,length(t_smooth),nptss)
        
        P2nrange_upper=zeros(3,nptss)
        llP2_upper=zeros(nptss)
        nllP2_upper=zeros(nptss)
        predict_P2_upper=zeros(2,length(t_smooth),nptss)
        predict_P2_realisations_upper_lq=zeros(2,length(t_smooth),nptss)
        predict_P2_realisations_upper_uq=zeros(2,length(t_smooth),nptss)

        # start at mle and increase parameter (upper)
        for i in 1:nptss
            function fun2(aa)
                return error(data,[aa[1],P2range_upper[i],aa[2],aa[3]])
            end
            lb1=[lb[1],lb[3],lb[4]];
            ub1=[ub[1],ub[3],ub[4]];
            local θG1=[P1mle, P3mle, P4mle]
            local (xo,fo)=optimise(fun2,θG1,lb1,ub1)
            P2nrange_upper[:,i]=xo[:]
            llP2_upper[i]=fo[1]

            modelmean = model(t_smooth,[P2nrange_upper[1,i],P2range_upper[i],P2nrange_upper[2,i],P2nrange_upper[3,i]])
            predict_P2_upper[:,:,i]=modelmean

            loop_data_dists=[Poisson(mi) for mi in modelmean]; 
            predict_P2_realisations_upper_lq[:,:,i]= [quantile(data_dist, THalpha/2) for data_dist in loop_data_dists];
            predict_P2_realisations_upper_uq[:,:,i]= [quantile(data_dist, 1-THalpha/2) for data_dist in loop_data_dists];

            if fo > fmle
                println("new MLE P2 upper")
                global fmle = fo
                global P1mle=P2nrange_upper[1,i]
                global P2mle=P2range_upper[i]
                global P3mle=P2nrange_upper[2,i]
                global P4mle=P2nrange_upper[3,i]
            end
        end
        
        # start at mle and decrease parameter (lower)
        for i in 1:nptss
            function fun2a(aa)
                return error(data,[aa[1],P2range_lower[i],aa[2],aa[3]])
            end
            lb1=[lb[1],lb[3],lb[4]];
            ub1=[ub[1],ub[3],ub[4]];
            local θG1=[P1mle, P3mle, P4mle]
            local (xo,fo)=optimise(fun2a,θG1,lb1,ub1)
            P2nrange_lower[:,i]=xo[:]
            llP2_lower[i]=fo[1]

            modelmean = model(t_smooth,[P2nrange_lower[1,i],P2range_lower[i],P2nrange_lower[2,i],P2nrange_lower[3,i]])
            predict_P2_lower[:,:,i]= modelmean

            loop_data_dists=[Poisson(mi) for mi in modelmean]; # normal distribution about mean
            predict_P2_realisations_lower_lq[:,:,i]= [quantile(data_dist, THalpha/2) for data_dist in loop_data_dists];
            predict_P2_realisations_lower_uq[:,:,i]= [quantile(data_dist, 1-THalpha/2) for data_dist in loop_data_dists];
            
            if fo > fmle
                println("new MLE S lower")
                global fmle = fo
                global P1mle=P2nrange_lower[1,i]
                global P2mle=P2range_lower[i]
                global P3mle=P2nrange_lower[2,i]
                global P4mle=P2nrange_lower[3,i]
            end
        end
        
        # combine the lower and upper
        P2range = [reverse(P2range_lower);P2range_upper]
        P2nrange = [reverse(P2nrange_lower); P2nrange_upper ]
        llP2 = [reverse(llP2_lower); llP2_upper]     
        nllP2=llP2.-maximum(llP2);
        
        max_P2=zeros(2,length(t_smooth))
        min_P2=100000*ones(2,length(t_smooth))
        for i in 1:(nptss)
            if (llP2_lower[i].-maximum(llP2)) >= TH
                for j in 1:length(t_smooth)
                    max_P2[1,j]=max(predict_P2_lower[1,j,i],max_P2[1,j])
                    min_P2[1,j]=min(predict_P2_lower[1,j,i],min_P2[1,j])
                    max_P2[2,j]=max(predict_P2_lower[2,j,i],max_P2[2,j])
                    min_P2[2,j]=min(predict_P2_lower[2,j,i],min_P2[2,j])
                end
            end
        end
        for i in 1:(nptss)
            if (llP2_upper[i].-maximum(llP2)) >= TH
                for j in 1:length(t_smooth)
                    max_P2[1,j]=max(predict_P2_upper[1,j,i],max_P2[1,j])
                    min_P2[1,j]=min(predict_P2_upper[1,j,i],min_P2[1,j]) 
                    max_P2[2,j]=max(predict_P2_upper[2,j,i],max_P2[2,j])
                    min_P2[2,j]=min(predict_P2_upper[2,j,i],min_P2[2,j])
                end
            end
        end
		 # combine the confidence sets for realisations
         max_realisations_P2=zeros(2,length(t_smooth))
         min_realisations_P2=1000*ones(2,length(t_smooth))
         for i in 1:(nptss)
             if (llP2_lower[i].-maximum(llP2)) >= TH_realisations
                 for j in 1:length(t_smooth)
                     max_realisations_P2[1,j]=max(predict_P2_realisations_lower_uq[1,j,i],max_realisations_P2[1,j])
                     min_realisations_P2[1,j]=min(predict_P2_realisations_lower_lq[1,j,i],min_realisations_P2[1,j])
                     max_realisations_P2[2,j]=max(predict_P2_realisations_lower_uq[2,j,i],max_realisations_P2[2,j])
                     min_realisations_P2[2,j]=min(predict_P2_realisations_lower_lq[2,j,i],min_realisations_P2[2,j])
                 end
             end
         end
         for i in 1:(nptss)
             if (llP2_upper[i].-maximum(llP2)) >= TH_realisations
                 for j in 1:length(t_smooth)
                     max_realisations_P2[1,j]=max(predict_P2_realisations_upper_uq[1,j,i],max_realisations_P2[1,j])
                     min_realisations_P2[1,j]=min(predict_P2_realisations_upper_lq[1,j,i],min_realisations_P2[1,j]) 
                     max_realisations_P2[2,j]=max(predict_P2_realisations_upper_uq[2,j,i],max_realisations_P2[2,j])
                     min_realisations_P2[2,j]=min(predict_P2_realisations_upper_lq[2,j,i],min_realisations_P2[2,j])
                 end
             end
         end

        #Profile V2
        P3min=lb[3]
        P3max=ub[3]
        P3range_lower=reverse(LinRange(P3min,P3mle,nptss))
        P3range_upper=LinRange(P3mle + (P3max-P3mle)/nptss,P3max,nptss)
        
        P3nrange_lower=zeros(3,nptss)
        llP3_lower=zeros(nptss)
        nllP3_lower=zeros(nptss)
        predict_P3_lower=zeros(2,length(t_smooth),nptss)
        predict_P3_realisations_lower_lq=zeros(2,length(t_smooth),nptss)
        predict_P3_realisations_lower_uq=zeros(2,length(t_smooth),nptss)

        P3nrange_upper=zeros(3,nptss)
        llP3_upper=zeros(nptss)
        nllP3_upper=zeros(nptss)
        predict_P3_upper=zeros(2,length(t_smooth),nptss)
        predict_P3_realisations_upper_lq=zeros(2,length(t_smooth),nptss)
        predict_P3_realisations_upper_uq=zeros(2,length(t_smooth),nptss)
        
        # start at mle and increase parameter (upper)
        for i in 1:nptss
            function fun3(aa)
                return error(data,[aa[1],aa[2],P3range_upper[i],aa[3]])
            end
            lb1=[lb[1],lb[2],lb[4]];
            ub1=[ub[1],ub[2],ub[4]];
            local θG1=[P1mle,P2mle,P4mle]    
            local (xo,fo)=optimise(fun3,θG1,lb1,ub1)
            P3nrange_upper[:,i]=xo[:]
            llP3_upper[i]=fo[1]

            modelmean = model(t_smooth,[P3nrange_upper[1,i],P3nrange_upper[2,i],P3range_upper[i],P3nrange_upper[3,i]])
            predict_P3_upper[:,:,i]= modelmean

            loop_data_dists=[Poisson(mi) for mi in modelmean]; # normal distribution about mean
            predict_P3_realisations_upper_lq[:,:,i]= [quantile(data_dist, THalpha/2) for data_dist in loop_data_dists];
            predict_P3_realisations_upper_uq[:,:,i]= [quantile(data_dist, 1-THalpha/2) for data_dist in loop_data_dists];


            if fo > fmle
                println("new MLE P3 upper")
                global fmle = fo
                global P1mle=P3nrange_upper[1,i]
                global P2mle=P3nrange_upper[2,i]
                global P3mle=P3range_upper[i]
                global P4mle=P3nrange_upper[3,i]
            end
        end
        
        # start at mle and decrease parameter (lower)
        for i in 1:nptss
            function fun3a(aa)
                return error(data,[aa[1],aa[2],P3range_lower[i],aa[3]])
            end
            lb1=[lb[1],lb[2],lb[4]];
            ub1=[ub[1],ub[2],ub[4]];
            local θG1=[P1mle,P2mle,P4mle]      
            local (xo,fo)=optimise(fun3a,θG1,lb1,ub1)
            P3nrange_lower[:,i]=xo[:]
            llP3_lower[i]=fo[1]

            modelmean = model(t_smooth,[P3nrange_lower[1,i],P3nrange_lower[2,i],P3range_lower[i],P3nrange_lower[3,i]])
            predict_P3_lower[:,:,i]=modelmean

            loop_data_dists=[Poisson(mi) for mi in modelmean]; # normal distribution about mean
            predict_P3_realisations_lower_lq[:,:,i]= [quantile(data_dist, THalpha/2) for data_dist in loop_data_dists];
            predict_P3_realisations_lower_uq[:,:,i]= [quantile(data_dist, 1-THalpha/2) for data_dist in loop_data_dists];


            if fo > fmle
                println("new MLE P3 lower")
                global fmle = fo
                global P1mle=P3nrange_lower[1,i]
                global P2mle=P3nrange_lower[2,i]
                global P3mle=P3range_lower[i]
                global P4mle=P3nrange_lower[3,i]
            end
        end
        
        # combine the lower and upper
        P3range = [reverse(P3range_lower);P3range_upper]
        P3nrange = [reverse(P3nrange_lower); P3nrange_upper ]
        llP3 = [reverse(llP3_lower); llP3_upper] 
        
        nllP3=llP3.-maximum(llP3)

        max_P3=zeros(2,length(t_smooth))
        min_P3=100000*ones(2,length(t_smooth))
        for i in 1:(nptss)
            if (llP3_lower[i].-maximum(llP3)) >= TH
                for j in 1:length(t_smooth)
                    max_P3[1,j]=max(predict_P3_lower[1,j,i],max_P3[1,j])
                    min_P3[1,j]=min(predict_P3_lower[1,j,i],min_P3[1,j])
                    max_P3[2,j]=max(predict_P3_lower[2,j,i],max_P3[2,j])
                    min_P3[2,j]=min(predict_P3_lower[2,j,i],min_P3[2,j])
                end
            end
        end
        for i in 1:(nptss)
            if (llP3_upper[i].-maximum(llP3)) >= TH
                for j in 1:length(t_smooth)
                    max_P3[1,j]=max(predict_P3_upper[1,j,i],max_P3[1,j])
                    min_P3[1,j]=min(predict_P3_upper[1,j,i],min_P3[1,j]) 
                    max_P3[2,j]=max(predict_P3_upper[2,j,i],max_P3[2,j])
                    min_P3[2,j]=min(predict_P3_upper[2,j,i],min_P3[2,j])
                end
            end
        end
         # combine the confidence sets for realisations
         max_realisations_P3=zeros(2,length(t_smooth))
         min_realisations_P3=1000*ones(2,length(t_smooth))
         for i in 1:(nptss)
             if (llP3_lower[i].-maximum(llP3)) >= TH_realisations
                 for j in 1:length(t_smooth)
                     max_realisations_P3[1,j]=max(predict_P3_realisations_lower_uq[1,j,i],max_realisations_P3[1,j])
                     min_realisations_P3[1,j]=min(predict_P3_realisations_lower_lq[1,j,i],min_realisations_P3[1,j])
                     max_realisations_P3[2,j]=max(predict_P3_realisations_lower_uq[2,j,i],max_realisations_P3[2,j])
                     min_realisations_P3[2,j]=min(predict_P3_realisations_lower_lq[2,j,i],min_realisations_P3[2,j])
                 end
             end
         end
         for i in 1:(nptss)
             if (llP3_upper[i].-maximum(llP3)) >= TH_realisations
                 for j in 1:length(t_smooth)
                     max_realisations_P3[1,j]=max(predict_P3_realisations_upper_uq[1,j,i],max_realisations_P3[1,j])
                     min_realisations_P3[1,j]=min(predict_P3_realisations_upper_lq[1,j,i],min_realisations_P3[1,j]) 
                     max_realisations_P3[2,j]=max(predict_P3_realisations_upper_uq[2,j,i],max_realisations_P3[2,j])
                     min_realisations_P3[2,j]=min(predict_P3_realisations_upper_lq[2,j,i],min_realisations_P3[2,j])
                 end
             end
         end
        
        #Profile K2
        P4min=lb[4]
        P4max=ub[4]
        P4range_lower=reverse(LinRange(P4min,P4mle,nptss))
        P4range_upper=LinRange(P4mle + (P4max-P4mle)/nptss,P4max,nptss)
        
        P4nrange_lower=zeros(3,nptss)
        llP4_lower=zeros(nptss)
        nllP4_lower=zeros(nptss)
        predict_P4_lower=zeros(2,length(t_smooth),nptss)
        predict_P4_realisations_lower_lq=zeros(2,length(t_smooth),nptss)
        predict_P4_realisations_lower_uq=zeros(2,length(t_smooth),nptss)
        
        P4nrange_upper=zeros(3,nptss)
        llP4_upper=zeros(nptss)
        nllP4_upper=zeros(nptss)
        predict_P4_upper=zeros(2,length(t_smooth),nptss)
        predict_P4_realisations_upper_lq=zeros(2,length(t_smooth),nptss)
        predict_P4_realisations_upper_uq=zeros(2,length(t_smooth),nptss)
        
        # start at mle and increase parameter (upper)
        for i in 1:nptss
            function fun4(aa)
                return error(data,[aa[1],aa[2],aa[3],P4range_upper[i]])
            end
            lb1=[lb[1],lb[2],lb[3]];
            ub1=[ub[1],ub[2],ub[3]];
            local θG1=[P1mle,P2mle,P3mle]
            local (xo,fo)=optimise(fun4,θG1,lb1,ub1)
            P4nrange_upper[:,i]=xo[:]
            llP4_upper[i]=fo[1]

            modelmean = model(t_smooth,[P4nrange_upper[1,i],P4nrange_upper[2,i],P4nrange_upper[3,i],P4range_upper[i]])
            predict_P4_upper[:,:,i]= modelmean

            loop_data_dists=[Poisson(mi) for mi in modelmean]; # normal distribution about mean
            predict_P4_realisations_upper_lq[:,:,i]= [quantile(data_dist, THalpha/2) for data_dist in loop_data_dists];
            predict_P4_realisations_upper_uq[:,:,i]= [quantile(data_dist, 1-THalpha/2) for data_dist in loop_data_dists];


            if fo > fmle
                println("new MLE P4 upper")
                global fmle = fo
                global P1mle=P4nrange_upper[1,i]
                global P2mle=P4nrange_upper[2,i]
                global P3mle=P4nrange_upper[3,i]
                global P4mle=P4range_upper[i]
            end
        end
        
        # start at mle and decrease parameter (lower)
        for i in 1:nptss
            function fun4a(aa)
                return error(data,[aa[1],aa[2],aa[3],P4range_lower[i]])
            end
            lb1=[lb[1],lb[2],lb[3]];
            ub1=[ub[1],ub[2],ub[3]];
            local θG1=[P1mle,P2mle,P3mle]
            local (xo,fo)=optimise(fun4a,θG1,lb1,ub1)
            P4nrange_lower[:,i]=xo[:]
            llP4_lower[i]=fo[1]

            modelmean =model(t_smooth,[P4nrange_lower[1,i],P4nrange_lower[2,i],P4nrange_lower[3,i],P4range_lower[i]])
            predict_P4_lower[:,:,i] = modelmean

            loop_data_dists=[Poisson(mi) for mi in modelmean]; 
            predict_P4_realisations_lower_lq[:,:,i]= [quantile(data_dist, THalpha/2) for data_dist in loop_data_dists];
            predict_P4_realisations_lower_uq[:,:,i]= [quantile(data_dist, 1-THalpha/2) for data_dist in loop_data_dists];


            if fo > fmle
                println("new MLE P4 lower")
                global fmle = fo
                global P1mle=P4nrange_lower[1,i]
                global P2mle=P4nrange_lower[2,i]
                global P3mle=P4nrange_lower[3,i]
                global P4mle=P4range_lower[i]
            end
        end
        
        # combine the lower and upper
        P4range = [reverse(P4range_lower);P4range_upper]
        llP4 = [reverse(llP4_lower); llP4_upper]        
        nllP4=llP4.-maximum(llP4);

        max_P4=zeros(2,length(t_smooth))
        min_P4=100000*ones(2,length(t_smooth))
        for i in 1:(nptss)
            if (llP4_lower[i].-maximum(llP4)) >= TH
                for j in 1:length(t_smooth)
                    max_P4[1,j]=max(predict_P4_lower[1,j,i],max_P4[1,j])
                    min_P4[1,j]=min(predict_P4_lower[1,j,i],min_P4[1,j])
                    max_P4[2,j]=max(predict_P4_lower[2,j,i],max_P4[2,j])
                    min_P4[2,j]=min(predict_P4_lower[2,j,i],min_P4[2,j])
                end
            end
        end
        for i in 1:(nptss)
            if (llP4_upper[i].-maximum(llP4)) >= TH
                for j in 1:length(t_smooth)
                    max_P4[1,j]=max(predict_P4_upper[1,j,i],max_P4[1,j])
                    min_P4[1,j]=min(predict_P4_upper[1,j,i],min_P4[1,j]) 
                    max_P4[2,j]=max(predict_P4_upper[2,j,i],max_P4[2,j])
                    min_P4[2,j]=min(predict_P4_upper[2,j,i],min_P4[2,j])
                end
            end
        end
        
       # combine the confidence sets for realisations
       max_realisations_P4=zeros(2,length(t_smooth))
       min_realisations_P4=1000*ones(2,length(t_smooth))
       for i in 1:(nptss)
           if (llP4_lower[i].-maximum(llP4)) >= TH_realisations
               for j in 1:length(t_smooth)
                   max_realisations_P4[1,j]=max(predict_P4_realisations_lower_uq[1,j,i],max_realisations_P4[1,j])
                   min_realisations_P4[1,j]=min(predict_P4_realisations_lower_lq[1,j,i],min_realisations_P4[1,j])
                   max_realisations_P4[2,j]=max(predict_P4_realisations_lower_uq[2,j,i],max_realisations_P4[2,j])
                   min_realisations_P4[2,j]=min(predict_P4_realisations_lower_lq[2,j,i],min_realisations_P4[2,j])
               end
           end
       end
       for i in 1:(nptss)
           if (llP4_upper[i].-maximum(llP4)) >= TH_realisations
               for j in 1:length(t_smooth)
                   max_realisations_P4[1,j]=max(predict_P4_realisations_upper_uq[1,j,i],max_realisations_P4[1,j])
                   min_realisations_P4[1,j]=min(predict_P4_realisations_upper_lq[1,j,i],min_realisations_P4[1,j]) 
                   max_realisations_P4[2,j]=max(predict_P4_realisations_upper_uq[2,j,i],max_realisations_P4[2,j])
                   min_realisations_P4[2,j]=min(predict_P4_realisations_upper_lq[2,j,i],min_realisations_P4[2,j])
               end
           end
       end

        ##############################################################################################################
        # Plot profile likelihoods
        
        combined_plot_linewidth = 2;

        # interpolate for smoother profile likelihoods
        interp_nptss= 1001;
        
        # V1
        interp_points_P1range =  LinRange(P1min,P1max,interp_nptss)
        interp_P1 = LinearInterpolation(P1range,nllP1)
        interp_nllP1 = interp_P1(interp_points_P1range)
        
        profile1=plot(interp_points_P1range,interp_nllP1,xlim=(P1min,P1max),ylim=(-4,0.1),yticks=[-3,-2,-1,0],xlab=L"V_{1}",ylab=L"\hat{\ell}_{p}",legend=false,lw=5,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=:deepskyblue3)
        profile1=hline!([-1.92],lw=2,linecolor=:black,linestyle=:dot)
        profile1=vline!([P1mle],lw=3,linecolor=:red)
        profile1=vline!([V1],lw=3,linecolor=:rosybrown,linestyle=:dash)

        # K1
        interp_points_P2range =  LinRange(P2min,P2max,interp_nptss)
        interp_P2 = LinearInterpolation(P2range,nllP2)
        interp_nllP2 = interp_P2(interp_points_P2range)
        
        profile2=plot(interp_points_P2range,interp_nllP2,xlim=(P2min,P2max),xticks=[0.0015, 0.0025, 0.0035],ylim=(-4,0.1),yticks=[-3,-2,-1,0],xlab=L"K_{1}",ylab=L"\hat{\ell}_{p}",legend=false,lw=5,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=:deepskyblue3)
        profile2=hline!([-1.92],lw=2,linecolor=:black,linestyle=:dot)
        profile2=vline!([P2mle],lw=3,linecolor=:red)
        profile2=vline!([K1],lw=3,linecolor=:rosybrown,linestyle=:dash)
        
        # V2
        interp_points_P3range =  LinRange(P3min,P3max,interp_nptss)
        interp_P3 = LinearInterpolation(P3range,nllP3)
        interp_nllP3 = interp_P3(interp_points_P3range)
        
        profile3=plot(interp_points_P3range,interp_nllP3,xlim=(P3min,P3max),xticks=[0.0015, 0.0025, 0.0035],ylim=(-4,0.1),yticks=[-3,-2,-1,0],xlab=L"V_{2}",ylab=L"\hat{\ell}_{p}",legend=false,lw=5,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=:deepskyblue3)
        profile3=hline!([-1.92],lw=2,linecolor=:black,linestyle=:dot)
        profile3=vline!([P3mle],lw=3,linecolor=:red)
        profile3=vline!([V2],lw=3,linecolor=:rosybrown,linestyle=:dash)


        # K2
        interp_points_P4range =  LinRange(P4min,P4max,interp_nptss)
        interp_P4 = LinearInterpolation(P4range,nllP4)
        interp_nllP4 = interp_P4(interp_points_P4range)
        
        profile4=plot(interp_points_P4range,interp_nllP4,xlim=(P4min,P4max),ylim=(-4,0.1),yticks=[-3,-2,-1,0],xlab=L"K_{2}",ylab=L"\hat{\ell}_{p}",legend=false,lw=5,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=:deepskyblue3)
        profile4=hline!([-1.92],lw=2,linecolor=:black,linestyle=:dot)
        profile4=vline!([P4mle],lw=3,linecolor=:red)
        profile4=vline!([K2],lw=3,linecolor=:rosybrown,linestyle=:dash)


        # Recompute best fit
        data0_smooth_MLE_recomputed = model(t_smooth,[P1mle,P2mle,P3mle,P4mle,C10,C20]); # generate data using known parameter values for plotting


        # plot model simulated at MLE and save
        p1_updated=plot(t_smooth,data0_smooth_MLE_recomputed[1,:],lw=3,linecolor=colour1) # solid - c1
        p1_updated=plot!(t_smooth,data0_smooth_MLE_recomputed[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
        p1_updated=scatter!(t,data[1,:],xlims=(0,t_max*1.1),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
        p1_updated=scatter!(t,data[2,:],xlims=(0,t_max*1.1),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,msw=0) # scatter c2
        p1_updated=hline!([0],lw=2,linecolor=:black,linestyle=:dash)

        display(p1_updated)
        savefig(p1_updated,filepath_save[1] * "Figp1_updated" * ".pdf")
            
        display(profile1)
        savefig(profile1,filepath_save[1] * "Fig_profile1"    * ".pdf")
        
        display(profile2)
        savefig(profile2,filepath_save[1] * "Fig_profile2"    * ".pdf")
        
        display(profile3)
        savefig(profile3,filepath_save[1] * "Fig_profile3"  * ".pdf")
        
        display(profile4)
        savefig(profile4,filepath_save[1] * "Fig_profile4"    * ".pdf")


        # plot residuals
        t_residuals = t;
        data0_MLE_recomputed_for_residuals = model(t,[P1mle,P2mle,P3mle,P4mle,C10,C20]); # generate data using known parameter values for plotting
        data_residuals = data-data0_MLE_recomputed_for_residuals
        maximum(abs.(data_residuals))

        p_residuals=scatter(t,data_residuals[1,:],xlims=(0,t_max),ylims=(-50,50),legend=false,markersize = 15,markercolor=colour1,markerstrokewidth=0) # scatter c1
        p_residuals=scatter!(t,data_residuals[2,:],xlims=(-1,t_max+1),ylims=(-50,50),legend=false,markersize =15,markercolor=colour2,xlab=L"t",ylab=L"\hat{e}_{i}",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,markerstrokewidth=0) # scatter c2        
        p_residuals=hline!([0],lw=2,linecolor=:black,linestyle=:dot)

        display(p_residuals)
        savefig(p_residuals,filepath_save[1] * "Figp_residuals"   * ".pdf")


     

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
        
        # V1
        (lb_CI_P1,ub_CI_P1) = fun_interpCI(P1mle,interp_points_P1range,interp_nllP1,TH)
        println(round(lb_CI_P1; digits = 4))
        println(round(ub_CI_P1; digits = 4))
        
        # K1
        (lb_CI_P2,ub_CI_P2) = fun_interpCI(P2mle,interp_points_P2range,interp_nllP2,TH)
        println(round(lb_CI_P2; digits = 3))
        println(round(ub_CI_P2; digits = 3))
        
        # V2
        (lb_CI_P3,ub_CI_P3) = fun_interpCI(P3mle,interp_points_P3range,interp_nllP3,TH)
        println(round(lb_CI_P3; digits = 3))
        println(round(ub_CI_P3; digits = 3))
        
        # K2
        (lb_CI_P4,ub_CI_P4) = fun_interpCI(P4mle,interp_points_P4range,interp_nllP4,TH)
        println(round(lb_CI_P4; digits = 3))
        println(round(ub_CI_P4; digits = 3))
        

        # Export MLE and bounds to csv (one file for all data) -- -MLE ONLY 
        if @isdefined(df_MLEBoundsAll) == 0
            println("not defined")
            global df_MLEBoundsAll = DataFrame(V1mle=P1mle, lb_CI_V1=lb_CI_P1, ub_CI_V1=ub_CI_P1, K1mle=P2mle, lb_CI_K1=lb_CI_P2, ub_CI_K1=ub_CI_P2, V2mle=P3mle, lb_CI_V2=lb_CI_P3, ub_CI_V2=ub_CI_P3, K2mle=P4mle, lb_CI_K2=lb_CI_P4, ub_CI_K2=ub_CI_P4 )
        else 
            println("DEFINED")
            global df_MLEBoundsAll_thisrow = DataFrame(V1mle=P1mle, lb_CI_V1=lb_CI_P1, ub_CI_V1=ub_CI_P1, K1mle=P2mle, lb_CI_K1=lb_CI_P2, ub_CI_K1=ub_CI_P2, V2mle=P3mle, lb_CI_V2=lb_CI_P3, ub_CI_V2=ub_CI_P3, K2mle=P4mle, lb_CI_K2=lb_CI_P4, ub_CI_K2=ub_CI_P4 )
            append!(df_MLEBoundsAll,df_MLEBoundsAll_thisrow)
        end
        
        CSV.write(filepath_save[1] * "MLEBoundsALL.csv", df_MLEBoundsAll)




#########################################################################################
        # Confidence sets for model solutions

           # P1
       
           ymle_smooth =  model(t_smooth,[P1mle,P2mle,P3mle,P4mle]); # MLE

           confmodel1=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1) # solid - c1
           confmodel1=plot!(t_smooth,ymle_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
           confmodel1=scatter!(t,data[1,:],xlims=(0,t_max+1),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
           confmodel1=scatter!(t,data[2,:],xlims=(0,t_max+1),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,msw=0) # scatter c2
           confmodel1=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_P1[1,:],max_P1[1,:].-ymle_smooth[1,:]),fillalpha=.2)
           confmodel1=plot!(t_smooth,ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_P1[2,:],max_P1[2,:].-ymle_smooth[2,:]),fillalpha=.2)
           confmodel1=hline!([0],lw=2,linecolor=:black,linestyle=:dash)

           display(confmodel1)
           savefig(confmodel1,filepath_save[1] * "Fig_confmodel1"   * ".pdf")
   
           # P2
           confmodel2=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1) # solid - c1
           confmodel2=plot!(t_smooth,ymle_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
           confmodel2=scatter!(t,data[1,:],xlims=(0,t_max+1),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
           confmodel2=scatter!(t,data[2,:],xlims=(0,t_max+1),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,msw=0) # scatter c2
           confmodel2=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_P2[1,:],max_P2[1,:].-ymle_smooth[1,:]),fillalpha=.2)
           confmodel2=plot!(t_smooth,ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_P2[2,:],max_P2[2,:].-ymle_smooth[2,:]),fillalpha=.2)
           confmodel2=hline!([0],lw=2,linecolor=:black,linestyle=:dash)

           display(confmodel2)
           savefig(confmodel2,filepath_save[1] * "Fig_confmodel2"   * ".pdf")

           # P3
           confmodel3=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1) # solid - c1
           confmodel3=plot!(t_smooth,ymle_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
           confmodel3=scatter!(t,data[1,:],xlims=(0,t_max+1),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
           confmodel3=scatter!(t,data[2,:],xlims=(0,t_max+1),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,msw=0) # scatter c2
           confmodel3=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_P3[1,:],max_P3[1,:].-ymle_smooth[1,:]),fillalpha=.2)
           confmodel3=plot!(t_smooth,ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_P3[2,:],max_P3[2,:].-ymle_smooth[2,:]),fillalpha=.2)
           confmodel3=hline!([0],lw=2,linecolor=:black,linestyle=:dash)

           display(confmodel3)
           savefig(confmodel3,filepath_save[1] * "Fig_confmodel3"   * ".pdf")
   

           # P4
           confmodel4=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1) # solid - c1
           confmodel4=plot!(t_smooth,ymle_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
           confmodel4=scatter!(t,data[1,:],xlims=(0,t_max+1),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
           confmodel4=scatter!(t,data[2,:],xlims=(0,t_max+1),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,msw=0) # scatter c2
           confmodel4=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_P4[1,:],max_P4[1,:].-ymle_smooth[1,:]),fillalpha=.2)
           confmodel4=plot!(t_smooth,ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_P4[2,:],max_P4[2,:].-ymle_smooth[2,:]),fillalpha=.2)
           confmodel4=hline!([0],lw=2,linecolor=:black,linestyle=:dash)

           display(confmodel4)
           savefig(confmodel4,filepath_save[1] * "Fig_confmodel4"   * ".pdf")
   
   
           # overall - plot model simulated at MLE, scatter data, and prediction interval.
           max_overall=zeros(2,length(t_smooth))
           min_overall=1000*ones(2,length(t_smooth))
           for j in 1:length(t_smooth)
               max_overall[1,j]=max(max_P1[1,j],max_P2[1,j],max_P3[1,j],max_P4[1,j])
               max_overall[2,j]=max(max_P1[2,j],max_P2[2,j],max_P3[2,j],max_P4[2,j])
               min_overall[1,j]=min(min_P1[1,j],min_P2[1,j],min_P3[1,j],min_P4[1,j])
               min_overall[2,j]=min(min_P1[2,j],min_P2[2,j],min_P3[2,j],min_P4[2,j])
           end
           confmodelu=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1) # solid - c1
           confmodelu=plot!(t_smooth,ymle_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
           confmodelu=scatter!(t,data[1,:],xlims=(0,t_max+1),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
           confmodelu=scatter!(t,data[2,:],xlims=(0,t_max+1),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,msw=0) # scatter c2 
           confmodelu=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_overall[1,:],max_overall[1,:].-ymle_smooth[1,:]),fillalpha=.2)
           confmodelu=plot!(t_smooth,ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_overall[2,:],max_overall[2,:].-ymle_smooth[2,:]),fillalpha=.2)
           confmodelu=hline!([0],lw=2,linecolor=:black,linestyle=:dash)

             display(confmodelu)
             savefig(confmodelu,filepath_save[1] * "Fig_confmodelu"   * ".pdf")



            
#########################################################################################
        # Plot difference of confidence set for model solutions and the model solution at MLE 

        ydiffmin=-100
        ydiffmax=100

        # P1
        confmodeldiff1=plot(t_smooth,0 .* ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_P1[1,:],max_P1[1,:].-ymle_smooth[1,:]),ylims=(ydiffmin,ydiffmax),legend=false,fillalpha=.2,lw=0,linecolor=colour1)
        confmodeldiff1=plot!(t_smooth,0 .* ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_P1[2,:],max_P1[2,:].-ymle_smooth[2,:]),fillalpha=.2,xlab=L"t",ylab=L"C_{y,0.95}^{V_{1}} - y(\hat{\theta})",lw=0,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2)
        confmodeldiff1=plot!(t_smooth,0 .*ymle_smooth[2,:],lw=1,c=:black,linestyle=:dash)
        
        display(confmodeldiff1)
        savefig(confmodeldiff1,filepath_save[1] * "Fig_confmodeldiff1"   * ".pdf")

        # P2 
        confmodeldiff2=plot(t_smooth,0 .* ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_P2[1,:],max_P2[1,:].-ymle_smooth[1,:]),ylims=(ydiffmin,ydiffmax),legend=false,fillalpha=.2,lw=0,linecolor=colour1)
        confmodeldiff2=plot!(t_smooth,0 .* ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_P2[2,:],max_P2[2,:].-ymle_smooth[2,:]),fillalpha=.2,xlab=L"t",ylab=L"C_{y,0.95}^{K_{1}} - y(\hat{\theta})",lw=0,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2)
        confmodeldiff2=plot!(t_smooth,0 .*ymle_smooth[2,:],lw=1,c=:black,linestyle=:dash)
        
        display(confmodeldiff2)
        savefig(confmodeldiff2,filepath_save[1] * "Fig_confmodeldiff2"   * ".pdf")

        # P3
        confmodeldiff3=plot(t_smooth,0 .* ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_P3[1,:],max_P3[1,:].-ymle_smooth[1,:]),ylims=(ydiffmin,ydiffmax),legend=false,fillalpha=.2,lw=0,linecolor=colour1)
        confmodeldiff3=plot!(t_smooth,0 .* ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_P3[2,:],max_P3[2,:].-ymle_smooth[2,:]),fillalpha=.2,xlab=L"t",ylab=L"C_{y,0.95}^{V_{2}} - y(\hat{\theta})",lw=0,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2)
        confmodeldiff3=plot!(t_smooth,0 .*ymle_smooth[2,:],lw=1,c=:black,linestyle=:dash)
        
        display(confmodeldiff3)
        savefig(confmodeldiff3,filepath_save[1] * "Fig_confmodeldiff3"   * ".pdf")

        # P4
        confmodeldiff4=plot(t_smooth,0 .* ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_P4[1,:],max_P4[1,:].-ymle_smooth[1,:]),ylims=(ydiffmin,ydiffmax),legend=false,fillalpha=.2,lw=0,linecolor=colour1)
        confmodeldiff4=plot!(t_smooth,0 .* ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_P4[2,:],max_P4[2,:].-ymle_smooth[2,:]),fillalpha=.2,xlab=L"t",ylab=L"C_{y,0.95}^{K_{2}} - y(\hat{\theta})",lw=0,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2)
        confmodeldiff4=plot!(t_smooth,0 .*ymle_smooth[2,:],lw=1,c=:black,linestyle=:dash)
        
        display(confmodeldiff4)
        savefig(confmodeldiff4,filepath_save[1] * "Fig_confmodeldiff4"   * ".pdf")
   

        # union 
       
        confmodeldiffu=plot(t_smooth,0 .* ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_overall[1,:],max_overall[1,:].-ymle_smooth[1,:]),ylims=(ydiffmin,ydiffmax),legend=false,fillalpha=.2,lw=0,linecolor=colour1)
        confmodeldiffu=plot!(t_smooth,0 .* ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_overall[2,:],max_overall[2,:].-ymle_smooth[2,:]),fillalpha=.2,xlab=L"t",ylab=L"C_{y,0.95} - y(\hat{\theta})",lw=0,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2)
        confmodeldiffu=plot!(t_smooth,0 .*ymle_smooth[2,:],lw=1,c=:black,linestyle=:dash)

        display(confmodeldiffu)
        savefig(confmodeldiffu,filepath_save[1] * "Fig_confmodeldiffu"   * ".pdf")


            
#########################################################################################
        # Confidence sets for model realisations
        
   xmin=0
   xmax=101
   ymin=-100
   ymax=410

   # P1
   confreal1=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1) # solid - c1
   confreal1=plot!(t_smooth,ymle_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
   confreal1=scatter!(t,data[1,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
   confreal1=scatter!(t,data[2,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,msw=0) # scatter c2
   confreal1=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_realisations_P1[1,:],max_realisations_P1[1,:].-ymle_smooth[1,:]),fillalpha=.2)
   confreal1=plot!(t_smooth,ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_realisations_P1[2,:],max_realisations_P1[2,:].-ymle_smooth[2,:]),fillalpha=.2)
   confreal1=hline!([0],lw=2,linecolor=:black,linestyle=:dash)
   
   display(confreal1)
   savefig(confreal1,filepath_save[1] * "Fig_confreal1"   * ".pdf")

   # P2
   confreal2=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1) # solid - c1
   confreal2=plot!(t_smooth,ymle_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
   confreal2=scatter!(t,data[1,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
   confreal2=scatter!(t,data[2,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,msw=0) # scatter c2
   confreal2=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_realisations_P2[1,:],max_realisations_P2[1,:].-ymle_smooth[1,:]),fillalpha=.2)
   confreal2=plot!(t_smooth,ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_realisations_P2[2,:],max_realisations_P2[2,:].-ymle_smooth[2,:]),fillalpha=.2)
   confreal2=hline!([0],lw=2,linecolor=:black,linestyle=:dash)

   display(confreal2)
   savefig(confreal2,filepath_save[1] * "Fig_confreal2"   * ".pdf")

   # P3
   confreal3=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1) # solid - c1
   confreal3=plot!(t_smooth,ymle_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
   confreal3=scatter!(t,data[1,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
   confreal3=scatter!(t,data[2,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,msw=0) # scatter c2
   confreal3=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_realisations_P3[1,:],max_realisations_P3[1,:].-ymle_smooth[1,:]),fillalpha=.2)
   confreal3=plot!(t_smooth,ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_realisations_P3[2,:],max_realisations_P3[2,:].-ymle_smooth[2,:]),fillalpha=.2)
   confreal3=hline!([0],lw=2,linecolor=:black,linestyle=:dash)
   
   display(confreal3)
   savefig(confreal3,filepath_save[1] * "Fig_confreal3"   * ".pdf")


    # P4
    confreal4=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1) # solid - c1
    confreal4=plot!(t_smooth,ymle_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
    confreal4=scatter!(t,data[1,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
    confreal4=scatter!(t,data[2,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,msw=0) # scatter c2
    confreal4=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_realisations_P4[1,:],max_realisations_P4[1,:].-ymle_smooth[1,:]),fillalpha=.2)
    confreal4=plot!(t_smooth,ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_realisations_P4[2,:],max_realisations_P4[2,:].-ymle_smooth[2,:]),fillalpha=.2)
    confreal4=hline!([0],lw=2,linecolor=:black,linestyle=:dash)
   

    display(confreal4)
    savefig(confreal4,filepath_save[1] * "Fig_confreal4"   * ".pdf")
   

     # union 
   max_realisations_overall=zeros(2,length(t_smooth))
   min_realisations_overall=1000*ones(2,length(t_smooth))
   for j in 1:length(t_smooth)
       max_realisations_overall[1,j]=max(max_realisations_P1[1,j],max_realisations_P2[1,j],max_realisations_P3[1,j],max_realisations_P4[1,j])
       max_realisations_overall[2,j]=max(max_realisations_P1[2,j],max_realisations_P2[2,j],max_realisations_P3[2,j],max_realisations_P4[2,j])
       min_realisations_overall[1,j]=min(min_realisations_P1[1,j],min_realisations_P2[1,j],min_realisations_P3[1,j],min_realisations_P4[1,j])
       min_realisations_overall[2,j]=min(min_realisations_P1[2,j],min_realisations_P2[2,j],min_realisations_P3[2,j],min_realisations_P4[2,j])
   end

   confrealu=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1) # solid - c1
   confrealu=plot!(t_smooth,ymle_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
   confrealu=scatter!(t,data[1,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
   confrealu=scatter!(t,data[2,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,msw=0) # scatter c2
   confrealu=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_realisations_overall[1,:],max_realisations_overall[1,:].-ymle_smooth[1,:]),fillalpha=.2)
   confrealu=plot!(t_smooth,ymle_smooth[2,:],w=0,c=colour2,ribbon=(ymle_smooth[2,:].-min_realisations_overall[2,:],max_realisations_overall[2,:].-ymle_smooth[2,:]),fillalpha=.2)
   confrealu=hline!([0],lw=2,linecolor=:black,linestyle=:dash)

   display(confrealu)
   savefig(confrealu,filepath_save[1] * "Fig_confrealu"   * ".pdf")


