# Murphy et al. (2023) 
# Implementing measurement error models with mechanistic mathematical models
# in a likelihood-based framework for estimation, identifiability analysis, 
# and prediction in the life sciences

# FigS7_DifferenceEquation_PoissonNoise_PoissonFit

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
isdir(pwd() * "\\FigS7_DifferenceEquation_PoissonNoise_PoissonFit\\") || mkdir(pwd() * "\\FigS7_DifferenceEquation_PoissonNoise_PoissonFit") # make folder to save figures if doesnt already exist
filepath_save = [pwd() * "\\FigS7_DifferenceEquation_PoissonNoise_PoissonFit\\"] # location to save figures "\\Fig2\\"] # location to save figures

#######################################################################################
## Mathematical model
a = zeros(2);
r1=0.1;
K1=100.0;
C10=5.0;
t_max = 80;
t=0:20:t_max # synthetic data measurement times
t_smooth = 0:(t_max+5); # for plotting known values and best-fits

#######################################################################################
## Solving the mathematical model

function model(t,a)
    r1=a[1];
    K1=a[2];
    tspanmax = Int(maximum(t))+1
    y_all=zeros(2,tspanmax)
    y_all[1,1] = C10
    for j=2:tspanmax
        y_all[1,j] = y_all[1,j-1]*exp(r1*(1 - y_all[1,j-1]/K1)) 
    end
    y=zeros(2,length(t))
    y[1,:] = y_all[1,(1 .+ t)]
    return y
end
    

#######################################################################################
## Generate Synthetic data

    Random.seed!(1)

    data0=0.0*zeros(1,length(t)); # generate vector to data from known parameters
    data=0.0*zeros(1,length(t)); # generate vector to data from known parameters + noise

    data0=model(t,[r1,K1]); # generate data using known parameter values to add noise
    data0[data0 .< 0] .= zero(eltype(data0))
    data0_smooth = model(t_smooth,[r1,K1]); # generate data using known parameter values for plotting

    data_dists=[Poisson(mi) for mi in data0] 
    data=[rand(data_dist) for data_dist in data_dists]; # generate the noisy data using Poisson

    global df_data = DataFrame(c1 = data[1,:], c2=data[2,:])
    CSV.write(filepath_save[1] * "Data.csv", df_data)


#######################################################################################
### Plot data - scatter plot

    ymax=150;

    colour1=:green2

    p0=plot(t_smooth,data0_smooth[1,:],lw=3,linecolor=colour1,xlab=L"t",ylab=L"N_{t}",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt) # solid - c1
    p0=scatter!(t,data[1,:],xlims=(0,t_max*1.1),ylims=(0,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1

    display(p0)
    savefig(p0,filepath_save[1] * "Fig0" * ".pdf")

#######################################################################################
### Parameter identifiability analysis - MLE, profiles

    rr1 = r1*1.05;
    KK1 = K1*1.05;

    TH=-1.921; #95% confidence interval threshold (for confidence interval for model parameters, and confidence set for model solutions)
    TH_realisations = -2.51 # 97.5 confidence interval threshold (for model realisations)
    THalpha = 0.025; # 97.5% data realisations interval threshold

    function error(data,a)
        y=zeros(length(t))
        y=model(t,a);
        e=0;
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

    
    θG = [rr1,KK1] # first guess

    # lower and upper bounds for parameter estimation
    r1_lb = r1*0.5 #5.0
    r1_ub = r1*1.5#30.0
    K1_lb = K1*0.5 #5.0
    K1_ub = K1*1.5 #6.0

    lb=[r1_lb,K1_lb];
    ub=[r1_ub,K1_ub];

    # MLE optimisation
    (xopt,fopt)  = optimise(fun,θG,lb,ub)

    # storing MLE 
    global fmle=fopt
    global r1mle=xopt[1]
    global K1mle=xopt[2]


    data0_smooth_MLE = model(t_smooth,[r1mle,K1mle]); # generate data using known parameter values for plotting


    # plot model simulated at MLE and save
    p1=plot(t_smooth,data0_smooth_MLE[1,:],lw=3,linecolor=colour1,xlab=L"t",ylab=L"N_{t}",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt) # solid - c1
    p1=scatter!(t,data[1,:],xlims=(0,t_max*1.1),ylims=(0,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1 
    display(p1)
    savefig(p1,filepath_save[1] * "Fig1" * ".pdf")


        #######################################################################################
        # Profiling (using the MLE as the first guess at each point)

        # PARAMETERS (P1, P2) = (r1, K1)
        global P1mle = r1mle
        global P2mle = K1mle

       
        nptss=40;

        #Profile P1 = r1

        P1min=lb[1]
        P1max=ub[1]
        P1range_lower=reverse(LinRange(P1min,P1mle,nptss))
        P1range_upper=LinRange(P1mle + (P1max-P1mle)/nptss,P1max,nptss)
        
        P1nrange_lower=zeros(1,nptss)
        llP1_lower=zeros(nptss)
        nllP1_lower=zeros(nptss)
        predict_P1_lower=zeros(2,length(t_smooth),nptss)
        predict_P1_realisations_lower_lq=zeros(2,length(t_smooth),nptss)
        predict_P1_realisations_lower_uq=zeros(2,length(t_smooth),nptss)
        
        P1nrange_upper=zeros(1,nptss)
        llP1_upper=zeros(nptss)
        nllP1_upper=zeros(nptss)
        predict_P1_upper=zeros(2,length(t_smooth),nptss)
        predict_P1_realisations_upper_lq=zeros(2,length(t_smooth),nptss)
        predict_P1_realisations_upper_uq=zeros(2,length(t_smooth),nptss)
        
        # start at mle and increase parameter (upper)
        for i in 1:nptss
            function fun1(aa)
                return error(data,[P1range_upper[i],aa[1]])
            end
            lb1=[lb[2]];
            ub1=[ub[2]];
            local θG1=[P2mle]
            local (xo,fo)=optimise(fun1,θG1,lb1,ub1)
            P1nrange_upper[:,i]=xo[:]
            llP1_upper[i]=fo[1]
            
            modelmean = model(t_smooth,[P1range_upper[i],P1nrange_upper[1,i]])
            predict_P1_upper[:,:,i]=modelmean

            loop_data_dists=[Poisson(mi) for mi in modelmean];
            predict_P1_realisations_upper_lq[:,:,i]= [quantile(data_dist, THalpha/2) for data_dist in loop_data_dists];
            predict_P1_realisations_upper_uq[:,:,i]= [quantile(data_dist, 1-THalpha/2) for data_dist in loop_data_dists];
            

            if fo > fmle
                println("new MLE P1 upper")
                global fmle=fo
                global P1mle=P1range_upper[i]
                global P2mle=P1nrange_upper[1,i]
            end
        end
        
        # start at mle and decrease parameter (lower)
        for i in 1:nptss
            function fun1a(aa)
                return error(data,[P1range_lower[i],aa[1]])
            end
            lb1=[lb[2]];
            ub1=[ub[2]];
            local θG1=[P2mle]
            local (xo,fo)=optimise(fun1a,θG1,lb1,ub1)
            P1nrange_lower[:,i]=xo[:]
            llP1_lower[i]=fo[1]

            modelmean =  model(t_smooth,[P1range_lower[i],P1nrange_lower[1,i]])
            predict_P1_lower[:,:,i]= modelmean

            loop_data_dists=[Poisson(mi) for mi in modelmean]; # normal distribution about mean
            predict_P1_realisations_lower_lq[:,:,i]= [quantile(data_dist, THalpha/2) for data_dist in loop_data_dists];
            predict_P1_realisations_lower_uq[:,:,i]= [quantile(data_dist, 1-THalpha/2) for data_dist in loop_data_dists];


            if fo > fmle
                println("new MLE P1 lower")
                global fmle = fo
                global P1mle=P1range_lower[i]
                global P2mle=P1nrange_lower[1,i]
            end
        end
        
        # combine the lower and upper
        P1range = [reverse(P1range_lower); P1range_upper]
        P1nrange = [reverse(P1nrange_lower); P1nrange_upper ]
        llP1 = [reverse(llP1_lower); llP1_upper] 
        nllP1=llP1.-maximum(llP1);

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
                 end
            end
        end
        for i in 1:(nptss)
            if (llP1_upper[i].-maximum(llP1)) >= TH_realisations
                for j in 1:length(t_smooth)
                    max_realisations_P1[1,j]=max(predict_P1_realisations_upper_uq[1,j,i],max_realisations_P1[1,j])
                    min_realisations_P1[1,j]=min(predict_P1_realisations_upper_lq[1,j,i],min_realisations_P1[1,j]) 
                 end
            end
        end
        

        #Profile P2 = k1
        P2min=lb[2]
        P2max=ub[2]
        P2range_lower=reverse(LinRange(P2min,P2mle,nptss))
        P2range_upper=LinRange(P2mle + (P2max-P2mle)/nptss,P2max,nptss)

        
        P2nrange_lower=zeros(1,nptss)
        llP2_lower=zeros(nptss)
        nllP2_lower=zeros(nptss)
        predict_P2_lower=zeros(2,length(t_smooth),nptss)
        predict_P2_realisations_lower_lq=zeros(2,length(t_smooth),nptss)
        predict_P2_realisations_lower_uq=zeros(2,length(t_smooth),nptss)
        
        P2nrange_upper=zeros(1,nptss)
        llP2_upper=zeros(nptss)
        nllP2_upper=zeros(nptss)
        predict_P2_upper=zeros(2,length(t_smooth),nptss)
        predict_P2_realisations_upper_lq=zeros(2,length(t_smooth),nptss)
        predict_P2_realisations_upper_uq=zeros(2,length(t_smooth),nptss)

        # start at mle and increase parameter (upper)
        for i in 1:nptss
            function fun2(aa)
                return error(data,[aa[1],P2range_upper[i]])
            end
            lb1=[lb[1]];
            ub1=[ub[1]];
            local θG1=[P1mle]
            local (xo,fo)=optimise(fun2,θG1,lb1,ub1)
            P2nrange_upper[:,i]=xo[:]
            llP2_upper[i]=fo[1]
           
            modelmean = model(t_smooth,[P2nrange_upper[1,i],P2range_upper[i]])
            predict_P2_upper[:,:,i]=modelmean

            loop_data_dists=[Poisson(mi) for mi in modelmean]; # normal distribution about mean
            predict_P2_realisations_upper_lq[:,:,i]= [quantile(data_dist, THalpha/2) for data_dist in loop_data_dists];
            predict_P2_realisations_upper_uq[:,:,i]= [quantile(data_dist, 1-THalpha/2) for data_dist in loop_data_dists];

            if fo > fmle
                println("new MLE P2 upper")
                global fmle = fo
                global P1mle=P2nrange_upper[1,i]
                global P2mle=P2range_upper[i]
            end
        end
        
        # start at mle and decrease parameter (lower)
        for i in 1:nptss
            function fun2a(aa)
                return error(data,[aa[1],P2range_lower[i]])
            end
            lb1=[lb[1]];
            ub1=[ub[1]];
            local θG1=[P1mle]
            local (xo,fo)=optimise(fun2a,θG1,lb1,ub1)
            P2nrange_lower[:,i]=xo[:]
            llP2_lower[i]=fo[1]

            modelmean = model(t_smooth,[P2nrange_lower[1,i],P2range_lower[i]])
            predict_P2_lower[:,:,i]=modelmean

            loop_data_dists=[Poisson(mi) for mi in modelmean]; # normal distribution about mean
            predict_P2_realisations_lower_lq[:,:,i]= [quantile(data_dist, THalpha/2) for data_dist in loop_data_dists];
            predict_P2_realisations_lower_uq[:,:,i]= [quantile(data_dist, 1-THalpha/2) for data_dist in loop_data_dists];


            if fo > fmle
                println("new MLE S lower")
                global fmle = fo
                global P1mle=P2nrange_lower[1,i]
                global P2mle=P2range_lower[i]
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
                end
            end
        end
        for i in 1:(nptss)
            if (llP2_upper[i].-maximum(llP2)) >= TH_realisations
                for j in 1:length(t_smooth)
                    max_realisations_P2[1,j]=max(predict_P2_realisations_upper_uq[1,j,i],max_realisations_P2[1,j])
                    min_realisations_P2[1,j]=min(predict_P2_realisations_upper_lq[1,j,i],min_realisations_P2[1,j]) 
                end
            end
        end

        ##############################################################################################################
        # Plot Profile likelihoods
        
        combined_plot_linewidth = 2;

        # interpolate for smoother profile likelihoods
        interp_nptss= 1001;
        
        # r1
        interp_points_P1range =  LinRange(P1min,P1max,interp_nptss)
        interp_P1 = LinearInterpolation(P1range,nllP1)
        interp_nllP1 = interp_P1(interp_points_P1range)
        
        profile1=plot(interp_points_P1range,interp_nllP1,xlim=(P1min,P1max),ylim=(-4,0.1),yticks=[-3,-2,-1,0],xlab=L"r",ylab=L"\hat{\ell}_{p}",legend=false,lw=5,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=:deepskyblue3)
        profile1=hline!([-1.92],lw=2,linecolor=:black,linestyle=:dot)
        profile1=vline!([P1mle],lw=3,linecolor=:red)
        profile1=vline!([r1],lw=3,linecolor=:rosybrown,linestyle=:dash)

        # K1
        interp_points_P2range =  LinRange(P2min,P2max,interp_nptss)
        interp_P2 = LinearInterpolation(P2range,nllP2)
        interp_nllP2 = interp_P2(interp_points_P2range)
        
        profile2=plot(interp_points_P2range,interp_nllP2,xlim=(P2min,P2max),ylim=(-4,0.1),yticks=[-3,-2,-1,0],xlab=L"K",ylab=L"\hat{\ell}_{p}",legend=false,lw=5,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=:deepskyblue3)
        profile2=hline!([-1.92],lw=2,linecolor=:black,linestyle=:dot)
        profile2=vline!([P2mle],lw=3,linecolor=:red)
        profile2=vline!([K1],lw=3,linecolor=:rosybrown,linestyle=:dash)

        # Recompute best fit
        data0_smooth_MLE_recomputed = model(t_smooth,[P1mle,P2mle]); # generate data using known parameter values for plotting

        # plot model simulated at MLE and save
        p1_updated=plot(t_smooth,data0_smooth_MLE_recomputed[1,:],lw=3,linecolor=colour1,xlab=L"t",ylab=L"N_{t}",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt) # solid - c1
        p1_updated=scatter!(t,data[1,:],xlims=(0,t_max*1.1),ylims=(0,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
         
        display(p1_updated)
        savefig(p1_updated,filepath_save[1] * "Figp1_updated" * ".pdf")
        
        display(profile1)
        savefig(profile1,filepath_save[1] * "Fig_profile1"    * ".pdf")
        
        display(profile2)
        savefig(profile2,filepath_save[1] * "Fig_profile2"    * ".pdf")
        
        # plot residuals
        t_residuals = t;
        data0_MLE_recomputed_for_residuals = model(t,[P1mle,P2mle]); # generate data using known parameter values for plotting
        data_residuals = data-data0_MLE_recomputed_for_residuals
        maximum(abs.(data_residuals))

        p_residuals=scatter(t,data_residuals[1,:],xlims=(-5,85),ylims=(-10,10),legend=false,markersize = 15,markercolor=colour1,markerstrokewidth=0,xlab=L"t",ylab=L"\hat{e}_{i}",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,msw=0) # scatter c1
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
              

        # Export MLE and bounds to csv (one file for all data) -- -MLE ONLY 
        if @isdefined(df_MLEBoundsAll) == 0
            println("not defined")
            global df_MLEBoundsAll = DataFrame(r1mle=P1mle, lb_CI_r1=lb_CI_P1, ub_CI_r1=ub_CI_P1, K1mle=P2mle, lb_CI_K1=lb_CI_P2, ub_CI_K1=ub_CI_P2)
        else 
            println("DEFINED")
            global df_MLEBoundsAll_thisrow = DataFrame(r1mle=P1mle, lb_CI_r1=lb_CI_P1, ub_CI_r1=ub_CI_P1, K1mle=P2mle, lb_CI_K1=lb_CI_P2, ub_CI_K1=ub_CI_P2)
            append!(df_MLEBoundsAll,df_MLEBoundsAll_thisrow)
        end
        
        CSV.write(filepath_save[1] * "MLEBoundsALL.csv", df_MLEBoundsAll)

#########################################################################################
        # Confidence sets for model solutions

           ymle_smooth =  model(t_smooth,[P1mle,P2mle]); # MLE
           
           # P1
           confmodel1=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1,xlab=L"t",ylab=L"N_{t}",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt) # solid - c1
           confmodel1=scatter!(t,data[1,:],xlims=(0,t_max+1),ylims=(0,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
           confmodel1=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_P1[1,:],max_P1[1,:].-ymle_smooth[1,:]),fillalpha=.2)

           display(confmodel1)
           savefig(confmodel1,filepath_save[1] * "Fig_confmodel1"   * ".pdf")
   
           # P2
           confmodel2=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1,xlab=L"t",ylab=L"N_{t}",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt) # solid - c1
           confmodel2=scatter!(t,data[1,:],xlims=(0,t_max+1),ylims=(0,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
           confmodel2=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_P2[1,:],max_P2[1,:].-ymle_smooth[1,:]),fillalpha=.2)
           
           display(confmodel2)
           savefig(confmodel2,filepath_save[1] * "Fig_confmodel2"   * ".pdf")


           # union
           max_overall=zeros(2,length(t_smooth))
           min_overall=1000*ones(2,length(t_smooth))
           for j in 1:length(t_smooth)
               max_overall[1,j]=max(max_P1[1,j],max_P2[1,j])
               max_overall[2,j]=max(max_P1[2,j],max_P2[2,j])
               min_overall[1,j]=min(min_P1[1,j],min_P2[1,j])
               min_overall[2,j]=min(min_P1[2,j],min_P2[2,j])
           end
           confmodelu=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1,xlab=L"t",ylab=L"N_{t}",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt) # solid - c1
           confmodelu=scatter!(t,data[1,:],xlims=(0,t_max+1),ylims=(0,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
           confmodelu=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_overall[1,:],max_overall[1,:].-ymle_smooth[1,:]),fillalpha=.2)
       
        display(confmodelu)
        savefig(confmodelu,filepath_save[1] * "Fig_confmodelu"   * ".pdf")
   
            
#########################################################################################
        # Plot difference of confidence set for model solutions and the model solution at MLE 

        ydiffmin=-20
        ydiffmax=20

        # P1
        confmodeldiff1=plot(t_smooth,0 .* ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_P1[1,:],max_P1[1,:].-ymle_smooth[1,:]),ylims=(ydiffmin,ydiffmax),legend=false,fillalpha=.2,lw=0,linecolor=colour1)
        confmodeldiff1=plot!(t_smooth,0 .*ymle_smooth[2,:],lw=1,c=:black,linestyle=:dash,xlab=L"t",ylab=L"C_{y,0.95}^{r} - y(\hat{\theta})",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt)
        
        display(confmodeldiff1)
        savefig(confmodeldiff1,filepath_save[1] * "Fig_confmodeldiff1"   * ".pdf")

        # P2 
        confmodeldiff2=plot(t_smooth,0 .* ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_P2[1,:],max_P2[1,:].-ymle_smooth[1,:]),ylims=(ydiffmin,ydiffmax),legend=false,fillalpha=.2,lw=0,linecolor=colour1)
        confmodeldiff2=plot!(t_smooth,0 .*ymle_smooth[2,:],lw=1,c=:black,linestyle=:dash,xlab=L"t",ylab=L"C_{y,0.95}^{K} - y(\hat{\theta})",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt)
        
        display(confmodeldiff2)
        savefig(confmodeldiff2,filepath_save[1] * "Fig_confmodeldiff2"   * ".pdf")

     
        # union 
        confmodeldiffu=plot(t_smooth,0 .* ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_overall[1,:],max_overall[1,:].-ymle_smooth[1,:]),ylims=(ydiffmin,ydiffmax),legend=false,fillalpha=.2,lw=0,linecolor=colour1)
        confmodeldiffu=plot!(t_smooth,0 .*ymle_smooth[2,:],lw=1,c=:black,linestyle=:dash,xlab=L"t",ylab=L"C_{y,0.95} - y(\hat{\theta})",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt)

        display(confmodeldiffu)
        savefig(confmodeldiffu,filepath_save[1] * "Fig_confmodeldiffu"   * ".pdf")


#########################################################################################
        # Confidence sets for model realisations
        
   xmin=0
   xmax=t_max*1.1
   ymin=-50
   ymax=ymax


   # P1
   confreal1=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1,xlab=L"t",ylab=L"N_{t}",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt) # solid - c1
   confreal1=scatter!(t,data[1,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
   confreal1=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_realisations_P1[1,:],max_realisations_P1[1,:].-ymle_smooth[1,:]),fillalpha=.2)
   confreal1=hline!([0],lw=2,linecolor=:black,linestyle=:dash)
   
   display(confreal1)
   savefig(confreal1,filepath_save[1] * "Fig_confreal1"   * ".pdf")

   # P2
   confreal2=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1,xlab=L"t",ylab=L"N_{t}",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt) # solid - c1
   confreal2=scatter!(t,data[1,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
   confreal2=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_realisations_P2[1,:],max_realisations_P2[1,:].-ymle_smooth[1,:]),fillalpha=.2)
   confreal2=hline!([0],lw=2,linecolor=:black,linestyle=:dash)

   display(confreal2)
   savefig(confreal2,filepath_save[1] * "Fig_confreal2"   * ".pdf")

   # union 
   max_realisations_overall=zeros(2,length(t_smooth))
   min_realisations_overall=1000*ones(2,length(t_smooth))
   for j in 1:length(t_smooth)
       max_realisations_overall[1,j]=max(max_realisations_P1[1,j],max_realisations_P2[1,j])
       min_realisations_overall[1,j]=min(min_realisations_P1[1,j],min_realisations_P2[1,j])
   end

   confrealu=plot(t_smooth,ymle_smooth[1,:],lw=3,linecolor=colour1,xlab=L"t",ylab=L"N_{t}",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt) # solid - c1
   confrealu=scatter!(t,data[1,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
   confrealu=plot!(t_smooth,ymle_smooth[1,:],w=0,c=colour1,ribbon=(ymle_smooth[1,:].-min_realisations_overall[1,:],max_realisations_overall[1,:].-ymle_smooth[1,:]),fillalpha=.2)
   confrealu=hline!([0],lw=2,linecolor=:black,linestyle=:dash)

   display(confrealu)
   savefig(confrealu,filepath_save[1] * "Fig_confrealu"   * ".pdf")


   