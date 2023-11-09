# Murphy et al. (2023) 
# Implementing measurement error models with mechanistic mathematical models
# in a likelihood-based framework for estimation, identifiability analysis, 
# and prediction in the life sciences

# Fig7_PDE_PoissonNoise_PoissonFit

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
using SpecialFunctions


pyplot() # plot options
fnt = Plots.font("sans-serif", 28) # plot options
global cur_colors = palette(:default) # plot options
isdir(pwd() * "\\Fig7_PDE_PoissonNoise_PoissonFit\\") || mkdir(pwd() * "\\Fig7_PDE_PoissonNoise_PoissonFit") # make folder to save figures if doesnt already exist
filepath_save = [pwd() * "\\Fig7_PDE_PoissonNoise_PoissonFit\\"] # location to save figures "\\Fig2\\"] # location to save figures

#######################################################################################
## Mathematical model

a = zeros(3); # three parameters, D, r1, r2
D=0.5;
r1=1.2;
r2=0.8;
t_record = [0.001, 0.25, 0.5, 0.75, 1.0]
L = 5; # half of domain length
h=1;
discrete_x = LinRange(-L,L,201)
discrete_t = t_record;


#######################################################################################
## Solving the mathematical model

function model(t_record,a)
    # PDE 
    D=a[1];
    r1=a[2];
    r2=a[3];
    function solu_analytical(x,t,h,D,r1,r2)
        out = 0.5*100*( erf.((h .- x) ./(2*sqrt(D*t))) + erf.((h .+ x)./(2*sqrt(D*t))) ) .* exp(-r1*t);
        return out
    end
    function solv_analytical(x,t,h,D,r1,r2)
        out = (r1/(r2-r1))*0.5*100*(erf.((h .- x)./(2*sqrt(D*t))) + erf.((h .+x)./(2*sqrt(D*t)))) .* (exp(-r1*t) .- exp(-r2*t));
        return(out)
    end
    modelsol = zeros(2,length(discrete_x),length(t_record))
    for i=1:length(t_record)
        modelsol[1,:,i] = solu_analytical(discrete_x,t_record[i],h,D,r1,r2);
        modelsol[2,:,i] = solv_analytical(discrete_x,t_record[i],h,D,r1,r2)
    end
    return modelsol
end


sol = model(t_record,[D,r1,r2])

solu = sol[1,:,:]
solv = sol[2,:,:]

### Plot exact data 

colour1=:green2
colour2=:magenta

anim = @animate for k in 1:length(discrete_t)
    plot(discrete_x,solu[1:end,k], title="t = $(discrete_t[k])",lw=3,linecolor=colour1) # 2:end since end = 1, periodic condition
    plot!(discrete_x,solv[1:end,k], title="t = $(discrete_t[k])",xlab=L"x",ylab=L"c_{1}(x,t), c_{2}(x,t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2,ylims=(0,105)) # 2:end since end = 1, periodic condition
end
gif(anim, filepath_save[1] * "Fig_anim" * ".gif", fps = 1)


### Plots
for k in 1:length(discrete_t)
    p0 = plot(discrete_x,solu[1:end,k], title="t = $(discrete_t[k])",lw=3,linecolor=colour1) # 2:end since end = 1, periodic condition
    p0 = plot!(p0,discrete_x,solv[1:end,k],xlab=L"x",ylab=L"c_{1}(x,t), c_{2}(x,t)",lw=3,legend=false,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2,ylim=(0,105)) # 2:end since end = 1, periodic condition
    display(p0)
    savefig(p0,filepath_save[1] * "Fig0_loop" * string(k) * ".pdf")
end


#######################################################################################
## Generate Synthetic data

    # for each time point - x measurements at -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5
    # time points 0.01, 0.25, 0.50, 0.75, 1.00
    Random.seed!(1) # random seed

    # [u for x at t0, x at t1, 
    # v for x at t0, v for x at t1]
    x_indexstore = [51,61,71,81,91,101,111,121,131,141,151];
    discrete_x_test = discrete_x[x_indexstore]
    x_record = discrete_x_test

    t_length = length(t_record)
    x_length = length(x_record)

    data0=0.0*zeros(2,t_length*x_length); # generate vector to data from known parameters
    data=0.0*zeros(2,t_length*x_length); # generate vector to data from known parameters + noise

    data0_fullsol=model(t_record,[D,r1,r2]); # generate data using known parameter values to add noise
    data0_fullsolu = data0_fullsol[1,:,:]
    data0_fullsolv = data0_fullsol[2,:,:]
    data0_solu = data0_fullsolu[x_indexstore,:]
    data0_solv = data0_fullsolv[x_indexstore,:]
    data0 = [data0_solu; data0_solv];
    
    data_dists=[Poisson(mi) for mi in data0]; # generate distributions to sample to generate noisy data - using Poisson 
    data=[rand(data_dist) for data_dist in data_dists]; # generate the noisy data

    ####################################################################################
### Plot data - scatter plot

colour1=:green2
colour2=:magenta

anim_withdata = @animate for k in 1:length(discrete_t)
    plot(discrete_x,solu[1:end,k], title="t = $(discrete_t[k])",lw=3,linecolor=colour1) # 2:end since end = 1, periodic condition
    plot!(discrete_x,solv[1:end,k], title="t = $(discrete_t[k])",xlab=L"x",ylab=L"c_{1}(x,t), c_{2}(x,t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # 2:end since end = 1, periodic condition
    # plot the data points
    scatter!(x_record,data[1:length(x_record),k],markersize = 8,markercolor=colour1,msw=0) # scatter c1
    scatter!(x_record,data[(length(x_record)+1):end,k],markersize = 8,markercolor=colour2,legend=false,ylim=(0,110),msw=0) # scatter c2
end
gif(anim_withdata, filepath_save[1] * "Fig_anim_withdata" * ".gif", fps = 1)


for k in 1:length(discrete_t)
    p0a = plot(discrete_x,solu[1:end,k], title="t = $(discrete_t[k])",lw=3,linecolor=colour1) # 2:end since end = 1, periodic condition
    p0a = plot!(p0a,discrete_x,solv[1:end,k], title="t = $(discrete_t[k])",xlab=L"x",ylab=L"c_{1}(x,t), c_{2}(x,t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # 2:end since end = 1, periodic condition
    # plot the data points
    p0a = scatter!(p0a,x_record,data[1:length(x_record),k],markersize = 8,markercolor=colour1,msw=0) # scatter c1
    p0a = scatter!(p0a,x_record,data[(length(x_record)+1):end,k],markersize = 8,markercolor=colour2,legend=false,ylim=(0,120),msw=0) # scatter c2
    display(p0a)
    savefig(p0a, filepath_save[1] * "Figp0a_loop" * string(k) * ".pdf")
end

#######################################################################################
### Parameter identifiability analysis - MLE, profiles

    # initial guesses for MLE search
    DD = 0.4;
    rr1=0.5;
    rr2=1.0;
    
    TH=-1.921; #95% confidence interval threshold (for confidence interval for model parameters, and confidence set for model solutions)
    TH_realisations = -2.51 # 97.5 confidence interval threshold (for model realisations)
    THalpha = 0.025; # 97.5% data realisations interval threshold

    function error(data,a)
        D=a[1];
        r1=a[2];
        r2=a[3];
        data0_fullsol=model(t_record,[D,r1,r2]); # generate data using known parameter values to add noise
        data0_fullsolu = data0_fullsol[1,:,:]
        data0_fullsolv = data0_fullsol[2,:,:]
        data0_solu = data0_fullsolu[x_indexstore,:]
        data0_solv = data0_fullsolv[x_indexstore,:]
        data0 = [data0_solu; data0_solv];
        data_dists=[Poisson(mi) for mi in data0]; # generate distributions to sample to generate noisy data - using Poisson 
        e=0;
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
        opt.maxtime = 20.0; # maximum time in seconds
        res =   NLopt.optimize(opt,θ₀)
        return res[[2,1]]
    end

    #######################################################################################
    # MLE

    θG = [DD,rr1,rr2] # first guess
    # lower and upper bounds for parameter estimation
    D_lb = 0.01
    D_ub = 1.01
    r1_lb = 0.5
    r1_ub = 1.5
    r2_lb = 0.5
    r2_ub = 1.5

    lb=[D_lb,r1_lb,r2_lb];
    ub=[D_ub,r1_ub,r2_ub];

    # MLE optimisation
    (xopt,fopt)  = optimise(fun,θG,lb,ub)

    # storing MLE 
    global fmle=fopt
    global Dmle=xopt[1]
    global r1mle=xopt[2]
    global r2mle=xopt[3]

    sol_MLE = model(t_record,[Dmle,r1mle,r2mle]) 
    sol_MLEu = sol_MLE[1,:,:]
    sol_MLEv = sol_MLE[2,:,:]

    colour1=:green2
    colour2=:magenta

    anim1_withdata = @animate for k in 1:length(discrete_t)
        plot(discrete_x,sol_MLEu[1:end,k], title="t = $(discrete_t[k])",lw=3,linecolor=colour1) # 2:end since end = 1, periodic condition
        plot!(discrete_x,sol_MLEv[1:end,k], title="t = $(discrete_t[k])",xlab=L"x",ylab=L"c_{1}(x,t), c_{2}(x,t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2,ylims=(0,105)) # 2:end since end = 1, periodic condition
        # plot the data points
        scatter!(x_record,data[1:length(x_record),k],markersize = 8,markercolor=colour1,msw=0) # scatter c1
        scatter!(x_record,data[(length(x_record)+1):end,k],markersize = 8,markercolor=colour2,legend=false,msw=0) # scatter c2
    end
    gif(anim1_withdata, filepath_save[1] * "Fig_1_anim_withdata" * ".gif", fps = 1)


    for k in 1:length(discrete_t)
        p1a = plot(discrete_x,sol_MLEu[1:end,k], title="t = $(discrete_t[k])",lw=3,linecolor=colour1) # 2:end since end = 1, periodic condition
        p1a = plot!(p1a,discrete_x,sol_MLEv[1:end,k], title="t = $(discrete_t[k])",xlab=L"x",ylab=L"c_{1}(x,t), c_{2}(x,t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2,ylims=(0,105)) # 2:end since end = 1, periodic condition
        # plot the data points
        p1a = scatter!(p1a,x_record,data[1:length(x_record),k],markersize = 8,markercolor=colour1,msw=0) # scatter c1
        p1a = scatter!(p1a,x_record,data[(length(x_record)+1):end,k],markersize = 8,markercolor=colour2,legend=false,msw=0) # scatter c2
        display(p1a)
        savefig(p1a, filepath_save[1] * "Figp1a_loop" * string(k) * ".pdf")
    end


#######################################################################################
# Profiling (using the MLE as the first guess at each point)

        # PARAMETERS (P1, P2, P3) = (D, r1, r2)
        global P1mle = Dmle
        global P2mle = r1mle
        global P3mle = r2mle
       
        nptss=40;

        #Profile D

        P1min=lb[1]
        P1max=ub[1]
        P1range_lower=reverse(LinRange(P1min,P1mle,nptss))
        P1range_upper=LinRange(P1mle + (P1max-P1mle)/nptss,P1max,nptss)
        
        P1nrange_lower=zeros(2,nptss)
        llP1_lower=zeros(nptss)
        nllP1_lower=zeros(nptss)
        predict_P1_lower=zeros(2,length(discrete_x),length(t_record),nptss)
        predict_P1_realisations_lower_lq=zeros(2,length(discrete_x),length(t_record),nptss)
        predict_P1_realisations_lower_uq=zeros(2,length(discrete_x),length(t_record),nptss)
        
        P1nrange_upper=zeros(2,nptss)
        llP1_upper=zeros(nptss)
        nllP1_upper=zeros(nptss)
        predict_P1_upper=zeros(2,length(discrete_x),length(t_record),nptss)
        predict_P1_realisations_upper_lq=zeros(2,length(discrete_x),length(t_record),nptss)
        predict_P1_realisations_upper_uq=zeros(2,length(discrete_x),length(t_record),nptss)

        # start at mle and increase parameter (upper)
        for i in 1:nptss
            function fun1(aa)
                return error(data,[P1range_upper[i],aa[1],aa[2]])
            end
            lb1=[lb[2],lb[3]];
            ub1=[ub[2],ub[3]];
            local θG1=[P2mle, P3mle]
            local (xo,fo)=optimise(fun1,θG1,lb1,ub1)
            P1nrange_upper[:,i]=xo[:]
            llP1_upper[i]=fo[1]

            modelmean = model(t_record,[P1range_upper[i],P1nrange_upper[1,i],P1nrange_upper[2,i]])  
            predict_P1_upper[:,:,:,i]=modelmean;

            loop_data_dists=[Poisson(mi) for mi in modelmean]; # normal distribution about mean
            predict_P1_realisations_upper_lq[:,:,:,i]= [quantile(data_dist, THalpha/2) for data_dist in loop_data_dists];
            predict_P1_realisations_upper_uq[:,:,:,i]= [quantile(data_dist, 1-THalpha/2) for data_dist in loop_data_dists];
            
            if fo > fmle
                println("new MLE P1 upper")
                global fmle=fo
                global P1mle=P1range_upper[i]
                global P2mle=P1nrange_upper[1,i]
                global P3mle=P1nrange_upper[2,i]
            end
        end
        
        # start at mle and decrease parameter (lower)
        for i in 1:nptss
            function fun1a(aa)
                return error(data,[P1range_lower[i],aa[1],aa[2]])
            end
            lb1=[lb[2],lb[3]];
            ub1=[ub[2],ub[3]];
            local θG1=[P2mle, P3mle]
            local (xo,fo)=optimise(fun1a,θG1,lb1,ub1)
            P1nrange_lower[:,i]=xo[:]
            llP1_lower[i]=fo[1]

            modelmean = model(t_record,[P1range_lower[i],P1nrange_lower[1,i],P1nrange_lower[2,i]])  
            predict_P1_lower[:,:,:,i]=modelmean;

            loop_data_dists=[Poisson(mi) for mi in modelmean]; # normal distribution about mean
            predict_P1_realisations_lower_lq[:,:,:,i]= [quantile(data_dist, THalpha/2) for data_dist in loop_data_dists];
            predict_P1_realisations_lower_uq[:,:,:,i]= [quantile(data_dist, 1-THalpha/2) for data_dist in loop_data_dists];

            if fo > fmle
                println("new MLE P1 lower")
                global fmle = fo
                global P1mle=P1range_lower[i]
                global P2mle=P1nrange_lower[1,i]
                global P3mle=P1nrange_lower[2,i]        
            end
        end
        
        # combine the lower and upper
        P1range = [reverse(P1range_lower); P1range_upper]
        P1nrange = [reverse(P1nrange_lower); P1nrange_upper ]
        llP1 = [reverse(llP1_lower); llP1_upper] 
        nllP1=llP1.-maximum(llP1);
        # combine the confidence sets for realisations
        max_realisations_P1=zeros(2,length(discrete_x),length(t_record))
        min_realisations_P1=1000*ones(2,length(discrete_x),length(t_record))
        for i in 1:(nptss)
            if (llP1_lower[i].-maximum(llP1)) >= TH_realisations
                for j in 1:length(discrete_x)
                    for k in 1:length(t_record)
                        max_realisations_P1[1,j,k]=max(predict_P1_realisations_lower_uq[1,j,k,i],max_realisations_P1[1,j,k])
                        min_realisations_P1[1,j,k]=min(predict_P1_realisations_lower_lq[1,j,k,i],min_realisations_P1[1,j,k])
                        max_realisations_P1[2,j,k]=max(predict_P1_realisations_lower_uq[2,j,k,i],max_realisations_P1[2,j,k])
                        min_realisations_P1[2,j,k]=min(predict_P1_realisations_lower_lq[2,j,k,i],min_realisations_P1[2,j,k])
                    end
                end
            end
        end
        for i in 1:(nptss)
            if (llP1_upper[i].-maximum(llP1)) >= TH_realisations
                for j in 1:length(discrete_x)
                    for k in 1:length(t_record)
                        max_realisations_P1[1,j,k]=max(predict_P1_realisations_upper_uq[1,j,k,i],max_realisations_P1[1,j,k])
                        min_realisations_P1[1,j,k]=min(predict_P1_realisations_upper_lq[1,j,k,i],min_realisations_P1[1,j,k]) 
                        max_realisations_P1[2,j,k]=max(predict_P1_realisations_upper_uq[2,j,k,i],max_realisations_P1[2,j,k])
                        min_realisations_P1[2,j,k]=min(predict_P1_realisations_upper_lq[2,j,k,i],min_realisations_P1[2,j,k])
                    end
                end
            end
        end

        

        #Profile r1
        P2min=lb[2]
        P2max=ub[2]
        P2range_lower=reverse(LinRange(P2min,P2mle,nptss))
        P2range_upper=LinRange(P2mle + (P2max-P2mle)/nptss,P2max,nptss)
        
        P2nrange_lower=zeros(2,nptss)
        llP2_lower=zeros(nptss)
        nllP2_lower=zeros(nptss)
        predict_P2_lower=zeros(2,length(discrete_x),length(t_record),nptss)
        predict_P2_realisations_lower_lq=zeros(2,length(discrete_x),length(t_record),nptss)
        predict_P2_realisations_lower_uq=zeros(2,length(discrete_x),length(t_record),nptss)
        
        P2nrange_upper=zeros(2,nptss)
        llP2_upper=zeros(nptss)
        nllP2_upper=zeros(nptss)
        predict_P2_upper=zeros(2,length(discrete_x),length(t_record),nptss)
        predict_P2_realisations_upper_lq=zeros(2,length(discrete_x),length(t_record),nptss)
        predict_P2_realisations_upper_uq=zeros(2,length(discrete_x),length(t_record),nptss)
        
        # start at mle and increase parameter (upper)
        for i in 1:nptss
            function fun2(aa)
                return error(data,[aa[1],P2range_upper[i],aa[2]])
            end
            lb1=[lb[1],lb[3]];
            ub1=[ub[1],ub[3]];
            local θG1=[P1mle, P3mle]
            local (xo,fo)=optimise(fun2,θG1,lb1,ub1)
            P2nrange_upper[:,i]=xo[:]
            llP2_upper[i]=fo[1]
          
		    modelmean = model(t_record,[P2nrange_upper[1,i],P2range_upper[i],P2nrange_upper[2,i]])  
            predict_P2_upper[:,:,:,i]=modelmean;

            loop_data_dists=[Poisson(mi) for mi in modelmean]; # normal distribution about mean
            predict_P2_realisations_upper_lq[:,:,:,i]= [quantile(data_dist, THalpha/2) for data_dist in loop_data_dists];
            predict_P2_realisations_upper_uq[:,:,:,i]= [quantile(data_dist, 1-THalpha/2) for data_dist in loop_data_dists];

            if fo > fmle
                println("new MLE P2 upper")
                global fmle = fo
                global P1mle=P2nrange_upper[1,i]
                global P2mle=P2range_upper[i]
                global P3mle=P2nrange_upper[2,i]
            end
        end
        
        # start at mle and decrease parameter (lower)
        for i in 1:nptss
            function fun2a(aa)
                return error(data,[aa[1],P2range_lower[i],aa[2]])
            end
            lb1=[lb[1],lb[3]];
            ub1=[ub[1],ub[3]];
            local θG1=[P1mle, P3mle]
            local (xo,fo)=optimise(fun2a,θG1,lb1,ub1)
            P2nrange_lower[:,i]=xo[:]
            llP2_lower[i]=fo[1]

		    modelmean = model(t_record,[P2nrange_lower[1,i],P2range_lower[i],P2nrange_lower[2,i]])  
            predict_P2_lower[:,:,:,i]=modelmean;

            loop_data_dists=[Poisson(mi) for mi in modelmean]; # normal distribution about mean
            predict_P2_realisations_lower_lq[:,:,:,i]= [quantile(data_dist, THalpha/2) for data_dist in loop_data_dists];
            predict_P2_realisations_lower_uq[:,:,:,i]= [quantile(data_dist, 1-THalpha/2) for data_dist in loop_data_dists];

            if fo > fmle
                println("new MLE S lower")
                global fmle = fo
                global P1mle=P2nrange_lower[1,i]
                global P2mle=P2range_lower[i]
                global P3mle=P2nrange_lower[2,i]
            end
        end
        
        # combine the lower and upper
        P2range = [reverse(P2range_lower);P2range_upper]
        P2nrange = [reverse(P2nrange_lower); P2nrange_upper ]
        llP2 = [reverse(llP2_lower); llP2_upper]     
        nllP2=llP2.-maximum(llP2);
		# combine the confidence sets for realisations
        max_realisations_P2=zeros(2,length(discrete_x),length(t_record))
        min_realisations_P2=1000*ones(2,length(discrete_x),length(t_record))
        for i in 1:(nptss)
            if (llP2_lower[i].-maximum(llP2)) >= TH_realisations
                for j in 1:length(discrete_x)
                    for k in 1:length(t_record)
                        max_realisations_P2[1,j,k]=max(predict_P2_realisations_lower_uq[1,j,k,i],max_realisations_P2[1,j,k])
                        min_realisations_P2[1,j,k]=min(predict_P2_realisations_lower_lq[1,j,k,i],min_realisations_P2[1,j,k])
                        max_realisations_P2[2,j,k]=max(predict_P2_realisations_lower_uq[2,j,k,i],max_realisations_P2[2,j,k])
                        min_realisations_P2[2,j,k]=min(predict_P2_realisations_lower_lq[2,j,k,i],min_realisations_P2[2,j,k])
                    end
                end
            end
        end
        for i in 1:(nptss)
            if (llP2_upper[i].-maximum(llP2)) >= TH_realisations
                for j in 1:length(discrete_x)
                    for k in 1:length(t_record)
                        max_realisations_P2[1,j,k]=max(predict_P2_realisations_upper_uq[1,j,k,i],max_realisations_P2[1,j,k])
                        min_realisations_P2[1,j,k]=min(predict_P2_realisations_upper_lq[1,j,k,i],min_realisations_P2[1,j,k]) 
                        max_realisations_P2[2,j,k]=max(predict_P2_realisations_upper_uq[2,j,k,i],max_realisations_P2[2,j,k])
                        min_realisations_P2[2,j,k]=min(predict_P2_realisations_upper_lq[2,j,k,i],min_realisations_P2[2,j,k])
                    end
                end
            end
        end

        

        #Profile r2
        P3min=lb[3]
        P3max=ub[3]
        P3range_lower=reverse(LinRange(P3min,P3mle,nptss))
        P3range_upper=LinRange(P3mle + (P3max-P3mle)/nptss,P3max,nptss)
        
        P3nrange_lower=zeros(2,nptss)
        llP3_lower=zeros(nptss)
        nllP3_lower=zeros(nptss)
        predict_P3_lower=zeros(2,length(discrete_x),length(t_record),nptss)
        predict_P3_realisations_lower_lq=zeros(2,length(discrete_x),length(t_record),nptss)
        predict_P3_realisations_lower_uq=zeros(2,length(discrete_x),length(t_record),nptss)
        
        P3nrange_upper=zeros(2,nptss)
        llP3_upper=zeros(nptss)
        nllP3_upper=zeros(nptss)
        predict_P3_upper=zeros(2,length(discrete_x),length(t_record),nptss)
        predict_P3_realisations_upper_lq=zeros(2,length(discrete_x),length(t_record),nptss)
        predict_P3_realisations_upper_uq=zeros(2,length(discrete_x),length(t_record),nptss)
        
        # start at mle and increase parameter (upper)
        for i in 1:nptss
            function fun3(aa)
                return error(data,[aa[1],aa[2],P3range_upper[i]])
            end
            lb1=[lb[1],lb[2]];
            ub1=[ub[1],ub[2]];
            local θG1=[P1mle,P2mle]    
            local (xo,fo)=optimise(fun3,θG1,lb1,ub1)
            P3nrange_upper[:,i]=xo[:]
            llP3_upper[i]=fo[1]

		    modelmean = model(t_record,[P3nrange_upper[1,i],P3nrange_upper[2,i],P3range_upper[i]])  
            predict_P3_upper[:,:,:,i]=modelmean;

            loop_data_dists=[Poisson(mi) for mi in modelmean]; # normal distribution about mean
            predict_P3_realisations_upper_lq[:,:,:,i]= [quantile(data_dist, THalpha/2) for data_dist in loop_data_dists];
            predict_P3_realisations_upper_uq[:,:,:,i]= [quantile(data_dist, 1-THalpha/2) for data_dist in loop_data_dists];

            if fo > fmle
                println("new MLE P3 upper")
                global fmle = fo
                global P1mle=P3nrange_upper[1,i]
                global P2mle=P3nrange_upper[2,i]
                global P3mle=P3range_upper[i]
            end
        end
        
        # start at mle and decrease parameter (lower)
        for i in 1:nptss
            function fun3a(aa)
                return error(data,[aa[1],aa[2],P3range_lower[i]])
            end
            lb1=[lb[1],lb[2]];
            ub1=[ub[1],ub[2]];
            local θG1=[P1mle,P2mle]      
            local (xo,fo)=optimise(fun3a,θG1,lb1,ub1)
            P3nrange_lower[:,i]=xo[:]
            llP3_lower[i]=fo[1]

		    modelmean = model(t_record,[P3nrange_lower[1,i],P3nrange_lower[2,i],P3range_lower[i]])  
            predict_P3_lower[:,:,:,i]=modelmean;

            loop_data_dists=[Poisson(mi) for mi in modelmean]; # normal distribution about mean
            predict_P3_realisations_lower_lq[:,:,:,i]= [quantile(data_dist, THalpha/2) for data_dist in loop_data_dists];
            predict_P3_realisations_lower_uq[:,:,:,i]= [quantile(data_dist, 1-THalpha/2) for data_dist in loop_data_dists];

            if fo > fmle
                println("new MLE P3 lower")
                global fmle = fo
                global P1mle=P3nrange_lower[1,i]
                global P2mle=P3nrange_lower[2,i]
                global P3mle=P3range_lower[i]
            end
        end
        
        # combine the lower and upper
        P3range = [reverse(P3range_lower);P3range_upper]
        P3nrange = [reverse(P3nrange_lower); P3nrange_upper ]
        llP3 = [reverse(llP3_lower); llP3_upper] 
        
        nllP3=llP3.-maximum(llP3)
        # combine the confidence sets for realisations
        max_realisations_P3=zeros(2,length(discrete_x),length(t_record))
        min_realisations_P3=1000*ones(2,length(discrete_x),length(t_record))
        for i in 1:(nptss)
            if (llP3_lower[i].-maximum(llP3)) >= TH_realisations
                for j in 1:length(discrete_x)
                    for k in 1:length(t_record)
                        max_realisations_P3[1,j,k]=max(predict_P3_realisations_lower_uq[1,j,k,i],max_realisations_P3[1,j,k])
                        min_realisations_P3[1,j,k]=min(predict_P3_realisations_lower_lq[1,j,k,i],min_realisations_P3[1,j,k])
                        max_realisations_P3[2,j,k]=max(predict_P3_realisations_lower_uq[2,j,k,i],max_realisations_P3[2,j,k])
                        min_realisations_P3[2,j,k]=min(predict_P3_realisations_lower_lq[2,j,k,i],min_realisations_P3[2,j,k])
                    end
                end
            end
        end
        for i in 1:(nptss)
            if (llP3_upper[i].-maximum(llP3)) >= TH_realisations
                for j in 1:length(discrete_x)
                    for k in 1:length(t_record)
                        max_realisations_P3[1,j,k]=max(predict_P3_realisations_upper_uq[1,j,k,i],max_realisations_P3[1,j,k])
                        min_realisations_P3[1,j,k]=min(predict_P3_realisations_upper_lq[1,j,k,i],min_realisations_P3[1,j,k]) 
                        max_realisations_P3[2,j,k]=max(predict_P3_realisations_upper_uq[2,j,k,i],max_realisations_P3[2,j,k])
                        min_realisations_P3[2,j,k]=min(predict_P3_realisations_upper_lq[2,j,k,i],min_realisations_P3[2,j,k])
                    end
                end
            end
        end
        
      
    
        ##############################################################################################################
        # Plot Profile likelihoods
        combined_plot_linewidth = 2;

        # interpolate for smoother profile likelihoods
        interp_nptss= 1001;
        
        # D
        interp_points_P1range =  LinRange(P1min,P1max,interp_nptss)
        interp_P1 = LinearInterpolation(P1range,nllP1)
        interp_nllP1 = interp_P1(interp_points_P1range)
        
        profile1=plot(interp_points_P1range,interp_nllP1,xlim=(P1min,P1max),ylim=(-4,0.1),yticks=[-3,-2,-1,0],xlab=L"D",ylab=L"\hat{\ell}_{p}",legend=false,lw=5,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=:deepskyblue3)
        profile1=hline!([-1.92],lw=2,linecolor=:black,linestyle=:dot)
        profile1=vline!([P1mle],lw=3,linecolor=:red)
        profile1=vline!([D],lw=3,linecolor=:rosybrown,linestyle=:dash)


        # r1
        interp_points_P2range =  LinRange(P2min,P2max,interp_nptss)
        interp_P2 = LinearInterpolation(P2range,nllP2)
        interp_nllP2 = interp_P2(interp_points_P2range)
        
        profile2=plot(interp_points_P2range,interp_nllP2,xlim=(P2min,P2max),ylim=(-4,0.1),yticks=[-3,-2,-1,0],xlab=L"r_{1}",ylab=L"\hat{\ell}_{p}",legend=false,lw=5,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=:deepskyblue3)
        profile2=hline!([-1.92],lw=2,linecolor=:black,linestyle=:dot)
        profile2=vline!([P2mle],lw=3,linecolor=:red)
        profile2=vline!([r1],lw=3,linecolor=:rosybrown,linestyle=:dash)
        
        # r2
        
        interp_points_P3range =  LinRange(P3min,P3max,interp_nptss)
        interp_P3 = LinearInterpolation(P3range,nllP3)
        interp_nllP3 = interp_P3(interp_points_P3range)
        
        profile3=plot(interp_points_P3range,interp_nllP3,xlim=(P3min,P3max),ylim=(-4,0.1),yticks=[-3,-2,-1,0],xlab=L"r_{2}",ylab=L"\hat{\ell}_{p}",legend=false,lw=5,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=:deepskyblue3)
        profile3=hline!([-1.92],lw=2,linecolor=:black,linestyle=:dot)
        profile3=vline!([P3mle],lw=3,linecolor=:red)
        profile3=vline!([r2],lw=3,linecolor=:rosybrown,linestyle=:dash)
        
        
        display(profile1)
        savefig(profile1,filepath_save[1] * "Fig_profile1"    * ".pdf")
        
        display(profile2)
        savefig(profile2,filepath_save[1] * "Fig_profile2"    * ".pdf")
        
        display(profile3)
        savefig(profile3,filepath_save[1] * "Fig_profile3"  * ".pdf")
        
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
        
        # D
        (lb_CI_P1,ub_CI_P1) = fun_interpCI(P1mle,interp_points_P1range,interp_nllP1,TH)
        println(round(lb_CI_P1; digits = 4))
        println(round(ub_CI_P1; digits = 4))
        
        # r1
        (lb_CI_P2,ub_CI_P2) = fun_interpCI(P2mle,interp_points_P2range,interp_nllP2,TH)
        println(round(lb_CI_P2; digits = 3))
        println(round(ub_CI_P2; digits = 3))
        
        # r2
        (lb_CI_P3,ub_CI_P3) = fun_interpCI(P3mle,interp_points_P3range,interp_nllP3,TH)
        println(round(lb_CI_P3; digits = 3))
        println(round(ub_CI_P3; digits = 3))
    
        # Export MLE and bounds to csv (one file for all data) -- -MLE ONLY 
        if @isdefined(df_MLEBoundsAll) == 0
            println("not defined")
            global df_MLEBoundsAll = DataFrame(Dmle=P1mle, lb_CI_D=lb_CI_P1, ub_CI_D=ub_CI_P1, r1mle=P2mle, lb_CI_r1=lb_CI_P2, ub_CI_r1=ub_CI_P2, r2mle=P3mle, lb_CI_r2=lb_CI_P3, ub_CI_r2=ub_CI_P3)
        else 
            println("DEFINED")
            global df_MLEBoundsAll_thisrow = DataFrame(Dmle=P1mle, lb_CI_D=lb_CI_P1, ub_CI_D=ub_CI_P1, r1mle=P2mle, lb_CI_r1=lb_CI_P2, ub_CI_r1=ub_CI_P2, r2mle=P3mle, lb_CI_r2=lb_CI_P3, ub_CI_r2=ub_CI_P3)
            append!(df_MLEBoundsAll,df_MLEBoundsAll_thisrow)
        end
        
        CSV.write(filepath_save[1] * "MLEBoundsALL.csv", df_MLEBoundsAll)



   #########################################################################################
        # Confidence sets for model realisations
        
        xmin=0
        xmax=25
        ymin=0
        ymax=1100

        # union 
        max_realisations_overall=zeros(2,length(discrete_x),length(t_record))
        min_realisations_overall=1000*ones(2,length(discrete_x),length(t_record))
        for j in 1:length(discrete_x)
            for k in 1:length(t_record)
                max_realisations_overall[1,j,k]=max(max_realisations_P1[1,j,k],max_realisations_P2[1,j,k],max_realisations_P3[1,j,k])
                max_realisations_overall[2,j,k]=max(max_realisations_P1[2,j,k],max_realisations_P2[2,j,k],max_realisations_P3[2,j,k])
                min_realisations_overall[1,j,k]=min(min_realisations_P1[1,j,k],min_realisations_P2[1,j,k],min_realisations_P3[1,j,k])
                min_realisations_overall[2,j,k]=min(min_realisations_P1[2,j,k],min_realisations_P2[2,j,k],min_realisations_P3[2,j,k])
            end
        end
 
        for k in 1:length(t_record)

            confrealu = plot(discrete_x,sol_MLEu[1:end,k], title="t = $(discrete_t[k])",lw=3,linecolor=colour1) # 2:end since end = 1, periodic condition
            confrealu = plot!(confrealu,discrete_x,sol_MLEv[1:end,k], title="t = $(discrete_t[k])",xlab=L"x",ylab=L"c_{1}(x,t), c_{2}(x,t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2,ylims=(0,125)) # 2:end since end = 1, periodic condition
            # plot the data points
            confrealu = scatter!(confrealu,x_record,data[1:length(x_record),k],markersize = 8,markercolor=colour1,msw=0) # scatter c1
            confrealu = scatter!(confrealu,x_record,data[(length(x_record)+1):end,k],markersize = 8,markercolor=colour2,legend=false,msw=0) # scatter c2
            # confidence set for realisations
            confrealu=plot!(confrealu,discrete_x,sol_MLEu[1:end,k],w=0,c=colour1,ribbon=(sol_MLEu[1:end,k].-min_realisations_overall[1,:,k],max_realisations_overall[1,:,k].-sol_MLEu[1:end,k]),fillalpha=.4)
            confrealu=plot!(confrealu,discrete_x,sol_MLEv[1:end,k],w=0,c=colour2,ribbon=(sol_MLEv[1:end,k].-min_realisations_overall[2,:,k],max_realisations_overall[2,:,k].-sol_MLEv[1:end,k]),fillalpha=.4)
            
            display(confrealu)
            savefig(confrealu, filepath_save[1] * "Fig_confrealu_" * string(k) * ".pdf")
        end





   
            
             
             


