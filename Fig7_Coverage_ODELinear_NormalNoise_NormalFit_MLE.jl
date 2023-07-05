# Murphy et al. (2023) 
# Implementing measurement error models in a likelihood-based framework for 
# estimation, identifiability analysis, and prediction in the life sciences

# Fig7_Coverage_ODELinear_NormalNoise_NormalFit_MLE

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
fnt = Plots.font("sans-serif", 28) # plot options
fntcovpanel = Plots.font("sans-serif", 12) # plot options
global cur_colors = palette(:default) # plot options
isdir(pwd() * "\\Fig7_Coverage_ODELinear_NormalNoise_NormalFit_MLE\\") || mkdir(pwd() * "\\Fig7_Coverage_ODELinear_NormalNoise_NormalFit_MLE") # make folder to save figures if doesnt already exist
filepath_save = [pwd() * "\\Fig7_Coverage_ODELinear_NormalNoise_NormalFit_MLE\\"] # location to save figures "\\Fig2\\"] # location to save figures

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
r1_withinbounds = zeros(totalseeds)
r2_withinbounds =  zeros(totalseeds)

r1modelsoln_withinbounds =  zeros(totalseeds)
r2modelsoln_withinbounds =  zeros(totalseeds)
modelsoln_withinbounds =  zeros(totalseeds)

r1realisations_withinbounds =  zeros(totalseeds)
r2realisations_withinbounds =  zeros(totalseeds)
realisations_withinbounds =  zeros(totalseeds)

r1modelsoln_countwithinbounds =  zeros(totalseeds)
r2modelsoln_countwithinbounds =  zeros(totalseeds)
modelsoln_countwithinbounds =  zeros(totalseeds)

realisations_MLE_withinbounds =  zeros(totalseeds)

# time
r1modelsoln_withinbounds_time =  zeros(totalseeds,2,length(t_smooth))
r2modelsoln_withinbounds_time =  zeros(totalseeds,2,length(t_smooth))
modelsoln_withinbounds_time =  zeros(totalseeds,2,length(t_smooth))

r1realisations_withinbounds_time  =  zeros(totalseeds,2,length(t_smooth))
r2realisations_withinbounds_time =  zeros(totalseeds,2,length(t_smooth))
realisations_withinbounds_time =  zeros(totalseeds,2,length(t_smooth))

realisations_MLE_withinbounds_time =  zeros(totalseeds,2,length(t))

r1realisationst_withinbounds_time  =  zeros(totalseeds,2,length(t))
r2realisationst_withinbounds_time =  zeros(totalseeds,2,length(t))
realisationst_withinbounds_time =  zeros(totalseeds,2,length(t))

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

    data0=model(t,[r1,r2,C10,C20]); # generate data using known parameter values to add noise
    data0_smooth = model(t_smooth,[r1,r2,C10,C20]); # generate data using known parameter values for plotting

    if seednum == 1
        p0modeltrue=plot(t_smooth,data0_smooth[1,:],lw=3,linecolor=colour1,ylims=(0,120)) # solid - c1
        p0modeltrue=plot!(t_smooth,data0_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2,legend=false) # solid - c2
        savefig(p0modeltrue,filepath_save[1] * "Fig_" * "p0modeltrue_i"  * string(seednum) *  ".pdf")
    end

    # generate distributions to sample to generate noisy data
    data_dists=[Normal(mi,Sd) for mi in model(t,[r1,r2,C10,C20])]; # normal distribution - standard deviation 1
    data=[rand(data_dist) for data_dist in data_dists]; # generate the noisy data
    data[data .< 0] .= zero(eltype(data)) # set negative values to zero

    Random.seed!(seednum+10000) # set random seed
    ### generate testdata to test prediction interval
    testdata=[rand(data_dist) for data_dist in data_dists]; # generate the noisy data
    testdata[testdata .< 0] .= zero(eltype(testdata))  # set negative values to zero

    #######################################################################################
    ### Plot data - scatter plot

    xmin=-0.05;
    xmax= 2.1;
    ymin= 0;
    ymax= 120;

    if seednum == 1
        # plot data used to generate confidence set
        p0scatter=scatter(t,data[1,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,msw=0) # scatter c1
        p0scatter=scatter!(t,data[2,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,msw=0) # scatter c2
        savefig(p0scatter,filepath_save[1] * "Fig_" * "p0scatter_i"  * string(seednum) *  ".pdf")

        # plot testdata (data which we use to test the confidence set for the model realisations)
        p0scattertest=plot(t_smooth,data0_smooth[1,:],lw=3,linecolor=colour1,ylims=(0,120)) # solid - c1
        p0scattertest=plot!(t_smooth,data0_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2,legend=false) # solid - c2
        p0scattertest=scatter!(t,testdata[1,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=:orange,xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,msw=0) # scatter c1
        p0scattertest=scatter!(t,testdata[2,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=:blue,msw=0) # scatter c2
        savefig(p0scattertest,filepath_save[1] * "Fig_" * "p0scattertest_i"  * string(seednum) *  ".pdf")


    end

    #######################################################################################
    ### Parameter identifiability analysis - MLE, profiles
        # initial guesses for MLE search
        rr1=1.01;
        rr2=0.51;
    
        TH=-1.921; #95% confidence interval threshold
        THrealisations_profile = -2.51  #97.5% confidence interval threshold
        THrealisations_prediction = 0.025; # 97.5% data realisations interval threshold
        THrealisations_MLE_prediction=0.05; #95% confidence interval threshold

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

        data0_MLE = model(t,[r1mle,r2mle,C10,C20]); # generate data using known parameter values for plotting

        if seednum == 1
        # plot model simulated at MLE and save
        p1=plot(t_smooth,data0_smooth_MLE[1,:],lw=3,linecolor=colour1) # solid - c1
        p1=plot!(t_smooth,data0_smooth_MLE[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # solid - c2
        p1=scatter!(t,data[1,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour1,msw=0) # scatter c1
        p1=scatter!(t,data[2,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=colour2,msw=0) # scatter c2

        display(p1)
        savefig(p1,filepath_save[1] * "Fig_" * "p1_i"  * string(seednum) *  ".pdf")


        end

        ########################################################################################
        ############ Coverage - confidence set for data realisations (MLE+NOISE)

        # Confidence interval for realisations - plot
        min_realisations_MLE_overall=zeros(2,length(t_smooth))
        max_realisations_MLE_overall=zeros(2,length(t_smooth))
        loop_data_dists_MLE=[Normal(mi,Sd) for mi in model(t_smooth,[r1mle,r2mle])]; # normal distribution about mean
        min_realisations_MLE_overall[:,:]= [quantile(data_dist, THrealisations_MLE_prediction/2) for data_dist in loop_data_dists_MLE];
        max_realisations_MLE_overall[:,:]= [quantile(data_dist, 1-THrealisations_MLE_prediction/2) for data_dist in loop_data_dists_MLE];
    
        # Confidence interval for realisation - compare with data
        min_realisations_MLEd_overall=zeros(2,length(t))
        max_realisations_MLEd_overall=zeros(2,length(t))
        loop_data_dists_MLEd=[Normal(mi,Sd) for mi in model(t,[r1mle,r2mle])]; # normal distribution about mean
        min_realisations_MLEd_overall[:,:]= [quantile(data_dist, THrealisations_MLE_prediction/2) for data_dist in loop_data_dists_MLEd];
        max_realisations_MLEd_overall[:,:]= [quantile(data_dist, 1-THrealisations_MLE_prediction/2) for data_dist in loop_data_dists_MLEd];
    

        for i in 1:length(t)
            # data realisations - MLE
            if (testdata[1,i] .>= min_realisations_MLEd_overall[1,i]) .* (testdata[1,i] .<= max_realisations_MLEd_overall[1,i])
                realisations_MLE_withinbounds_time[seednum,1,i] = 1
            end
            if (testdata[2,i] .>= min_realisations_MLEd_overall[2,i]) .* (testdata[2,i] .<= max_realisations_MLEd_overall[2,i])
                realisations_MLE_withinbounds_time[seednum,2,i] = 1
            end
        end

        # is the true model solution within the model solution bounds from r1
        if (sum(testdata .>= min_realisations_MLEd_overall) == 2*length(t_smooth)) .* (sum(testdata .<= max_realisations_MLEd_overall) == 2*length(t_smooth))
            realisations_MLE_withinbounds[seednum] = 1
        end

        # plot
        if seednum == 1
            ytrue_smooth =  model(t_smooth,[r1,r2,C10,C20]); 
            # plot model simulated at MLE and save
            p1mlepredictionintervalonly=plot(t_smooth,data0_smooth_MLE[1,:],w=0,c=colour1,ribbon=(data0_smooth_MLE[1,:].-min_realisations_MLE_overall[1,:],max_realisations_MLE_overall[1,:].-data0_smooth_MLE[1,:]),fillalpha=.2,xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,legend=false) # Confidence set for data realisations - c1
            p1mlepredictionintervalonly=plot!(t_smooth,data0_smooth_MLE[2,:],w=0,c=colour2,ribbon=(data0_smooth_MLE[2,:].-min_realisations_MLE_overall[2,:],max_realisations_MLE_overall[2,:].-data0_smooth_MLE[2,:]),fillalpha=.2,xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,ylims=(0,120)) # Confidence set for data realisations - c2
            display(p1mlepredictionintervalonly)
            savefig(p1mlepredictionintervalonly,filepath_save[1] * "Fig_" * "p1mlepredictionintervalonly_i"  * string(seednum) *  ".pdf")
            
            p1mleprediction=plot(t_smooth,ytrue_smooth[1,:],lw=3,linecolor=:black) # MLE- solid - c1
            p1mleprediction=plot!(t_smooth,ytrue_smooth[2,:],xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",lw=3,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=:black) # MLE - solid - c2
            p1mleprediction=plot!(t_smooth,data0_smooth_MLE[1,:],w=0,c=colour1,ribbon=(data0_smooth_MLE[1,:].-min_realisations_MLE_overall[1,:],max_realisations_MLE_overall[1,:].-data0_smooth_MLE[1,:]),fillalpha=.2,xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,legend=false) # Confidence set for data realisations - c1
            p1mleprediction=plot!(t_smooth,data0_smooth_MLE[2,:],w=0,c=colour2,ribbon=(data0_smooth_MLE[2,:].-min_realisations_MLE_overall[2,:],max_realisations_MLE_overall[2,:].-data0_smooth_MLE[2,:]),fillalpha=.2,xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt) # Confidence set for data realisations - c2
            p1mleprediction=scatter!(t,testdata[1,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=:orange,msw=0) # Data - scatter c1
            p1mleprediction=scatter!(t,testdata[2,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=:blue,msw=0) # Data - scatter c2
            display(p1mleprediction)
            savefig(p1mleprediction,filepath_save[1] * "Fig_" * "p1mleprediction_i"  * string(seednum) *  ".pdf")

            
            p1mleprediction2=plot(t_smooth,data0_smooth_MLE[1,:],w=0,c=colour1,ribbon=(data0_smooth_MLE[1,:].-min_realisations_MLE_overall[1,:],max_realisations_MLE_overall[1,:].-data0_smooth_MLE[1,:]),fillalpha=.2,xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,legend=false) # Confidence set for data realisations - c1
            p1mleprediction2=plot!(t_smooth,data0_smooth_MLE[2,:],w=0,c=colour2,ribbon=(data0_smooth_MLE[2,:].-min_realisations_MLE_overall[2,:],max_realisations_MLE_overall[2,:].-data0_smooth_MLE[2,:]),fillalpha=.2,xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt) # Confidence set for data realisations - c2
            p1mleprediction2=scatter!(t,testdata[1,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=:orange,msw=0) # Data - scatter c1
            p1mleprediction2=scatter!(t,testdata[2,:],xlims=(xmin,xmax),ylims=(ymin,ymax),legend=false,markersize = 8,markercolor=:blue,msw=0) # Data - scatter c2
            display(p1mleprediction2)
            savefig(p1mleprediction2,filepath_save[1] * "Fig_" * "p1mleprediction2_i"  * string(seednum) *  ".pdf")
            
            # plot difference
            ytrue_smooth =  model(t_smooth,[r1,r2,C10,C20]); 
            ydiffmin=-20
            ydiffmax=20

            modelrealconfidenceset_MLE_truesoln_diff_c1=plot(t_smooth,0 .*data0_smooth_MLE[1,:],w=0,c=colour1,ribbon=(data0_smooth_MLE[1,:].-min_realisations_MLE_overall[1,:],max_realisations_MLE_overall[1,:].-data0_smooth_MLE[1,:]),ylims=(ydiffmin,ydiffmax),legend=false,fillalpha=.2,xlab=L"t",ylab=L"C_{z_{i},0.95} - y(\hat{\theta})",lw=0,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour1)
            modelrealconfidenceset_MLE_truesoln_diff_c1=plot!(t_smooth,ytrue_smooth[1,:] .- data0_smooth_MLE[1,:],lw=3,linecolor=:black)
            modelrealconfidenceset_MLE_truesoln_diff_c1=scatter!(t,data[1,:]-data0_MLE[1,:],legend=false,markersize = 8,markercolor=:orange,msw=0)
            display(modelrealconfidenceset_MLE_truesoln_diff_c1)
            savefig(modelrealconfidenceset_MLE_truesoln_diff_c1,filepath_save[1] * "Fig_" * "modelrealconfidenceset_MLE_truesoln_diff_c1"  * string(seednum) *  ".pdf")
        
            modelrealconfidenceset_MLE_truesoln_diff_c2=plot(t_smooth,0 .*data0_smooth_MLE[2,:],w=0,c=colour2,ribbon=(data0_smooth_MLE[2,:].-min_realisations_MLE_overall[2,:],max_realisations_MLE_overall[2,:].-data0_smooth_MLE[2,:]),ylims=(ydiffmin,ydiffmax),fillalpha=.2,xlab=L"t",ylab=L"C_{z_{i},0.95} - y(\hat{\theta})",lw=0,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2,legend=false)
            modelrealconfidenceset_MLE_truesoln_diff_c2=plot!(t_smooth,ytrue_smooth[2,:] .- data0_smooth_MLE[2,:],lw=3,linecolor=:black)
            modelrealconfidenceset_MLE_truesoln_diff_c2=scatter!(t,data[2,:]-data0_MLE[2,:],legend=false,markersize = 8,markercolor=:blue,msw=0)
            display(modelrealconfidenceset_MLE_truesoln_diff_c2)
            savefig(modelrealconfidenceset_MLE_truesoln_diff_c2,filepath_save[1] * "Fig_" * "modelrealconfidenceset_MLE_truesoln_diff_c2" * string(seednum) * ".pdf")
        end

    #     #######################################################################################
    #         # Profiling (using the MLE as the first guess at each point)
        
            nptss=20;

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
            predict_r1_realisationst_lower_lq=zeros(2,length(t),nptss)
            predict_r1_realisationst_lower_uq=zeros(2,length(t),nptss)
            
            nr1range_upper=zeros(1,nptss)
            llr1_upper=zeros(nptss)
            nllr1_upper=zeros(nptss)
            predict_r1_upper=zeros(2,length(t_smooth),nptss)
            predict_r1_realisations_upper_lq=zeros(2,length(t_smooth),nptss)
            predict_r1_realisations_upper_uq=zeros(2,length(t_smooth),nptss)
            predict_r1_realisationst_upper_lq=zeros(2,length(t),nptss)
            predict_r1_realisationst_upper_uq=zeros(2,length(t),nptss)

            
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
                predict_r1_realisations_upper_lq[:,:,i]= [quantile(data_dist, THrealisations_prediction/2) for data_dist in loop_data_dists];
                predict_r1_realisations_upper_uq[:,:,i]= [quantile(data_dist, 1-THrealisations_prediction/2) for data_dist in loop_data_dists];

                loop_data_distst=[Normal(mi,Sd) for mi in model(t,[r1range_upper[i],nr1range_upper[1,i]])]; # normal distribution about mean
                predict_r1_realisationst_upper_lq[:,:,i]= [quantile(data_dist, THrealisations_prediction/2) for data_dist in loop_data_distst];
                predict_r1_realisationst_upper_uq[:,:,i]= [quantile(data_dist, 1-THrealisations_prediction/2) for data_dist in loop_data_distst];
                if fo > fmle
                    # println("new MLE r1 upper")
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
                predict_r1_realisations_lower_lq[:,:,i]= [quantile(data_dist, THrealisations_prediction/2) for data_dist in loop_data_dists];
                predict_r1_realisations_lower_uq[:,:,i]= [quantile(data_dist, 1-THrealisations_prediction/2) for data_dist in loop_data_dists];

                loop_data_distst=[Normal(mi,Sd) for mi in model(t,[r1range_lower[i],nr1range_lower[1,i]])]; # normal distribution about mean
                predict_r1_realisationst_lower_lq[:,:,i]= [quantile(data_dist, THrealisations_prediction/2) for data_dist in loop_data_distst];
                predict_r1_realisationst_lower_uq[:,:,i]= [quantile(data_dist, 1-THrealisations_prediction/2) for data_dist in loop_data_distst];

                if fo > fmle
                    # println("new MLE r2 lower")
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
                if (llr1_lower[i].-maximum(llr1)) >= THrealisations_profile
                    for j in 1:length(t_smooth)
                        max_realisations_r1[1,j]=max(predict_r1_realisations_lower_uq[1,j,i],max_realisations_r1[1,j])
                        min_realisations_r1[1,j]=min(predict_r1_realisations_lower_lq[1,j,i],min_realisations_r1[1,j])
                        max_realisations_r1[2,j]=max(predict_r1_realisations_lower_uq[2,j,i],max_realisations_r1[2,j])
                        min_realisations_r1[2,j]=min(predict_r1_realisations_lower_lq[2,j,i],min_realisations_r1[2,j])
                    end
                end
            end
            for i in 1:(nptss)
                if (llr1_upper[i].-maximum(llr1)) >= THrealisations_profile
                    for j in 1:length(t_smooth)
                        max_realisations_r1[1,j]=max(predict_r1_realisations_upper_uq[1,j,i],max_realisations_r1[1,j])
                        min_realisations_r1[1,j]=min(predict_r1_realisations_upper_lq[1,j,i],min_realisations_r1[1,j]) 
                        max_realisations_r1[2,j]=max(predict_r1_realisations_upper_uq[2,j,i],max_realisations_r1[2,j])
                        min_realisations_r1[2,j]=min(predict_r1_realisations_upper_lq[2,j,i],min_realisations_r1[2,j])
                    end
                end
            end
              # combine the confidence sets for realisations
              max_realisationst_r1=zeros(2,length(t))
              min_realisationst_r1=1000*ones(2,length(t))
              for i in 1:(nptss)
                  if (llr1_lower[i].-maximum(llr1)) >= THrealisations_profile
                      for j in 1:length(t)
                          max_realisationst_r1[1,j]=max(predict_r1_realisationst_lower_uq[1,j,i],max_realisationst_r1[1,j])
                          min_realisationst_r1[1,j]=min(predict_r1_realisationst_lower_lq[1,j,i],min_realisationst_r1[1,j])
                          max_realisationst_r1[2,j]=max(predict_r1_realisationst_lower_uq[2,j,i],max_realisationst_r1[2,j])
                          min_realisationst_r1[2,j]=min(predict_r1_realisationst_lower_lq[2,j,i],min_realisationst_r1[2,j])
                      end
                  end
              end
              for i in 1:(nptss)
                  if (llr1_upper[i].-maximum(llr1)) >= THrealisations_profile
                      for j in 1:length(t)
                          max_realisationst_r1[1,j]=max(predict_r1_realisationst_upper_uq[1,j,i],max_realisationst_r1[1,j])
                          min_realisationst_r1[1,j]=min(predict_r1_realisationst_upper_lq[1,j,i],min_realisationst_r1[1,j]) 
                          max_realisationst_r1[2,j]=max(predict_r1_realisationst_upper_uq[2,j,i],max_realisationst_r1[2,j])
                          min_realisationst_r1[2,j]=min(predict_r1_realisationst_upper_lq[2,j,i],min_realisationst_r1[2,j])
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

            predict_r2_realisationst_lower_lq=zeros(2,length(t),nptss)
            predict_r2_realisationst_lower_uq=zeros(2,length(t),nptss)
            
            nr2range_upper=zeros(1,nptss)
            llr2_upper=zeros(nptss)
            nllr2_upper=zeros(nptss)
            predict_r2_upper=zeros(2,length(t_smooth),nptss)
            predict_r2_realisations_upper_lq=zeros(2,length(t_smooth),nptss)
            predict_r2_realisations_upper_uq=zeros(2,length(t_smooth),nptss)

            predict_r2_realisationst_upper_lq=zeros(2,length(t),nptss)
            predict_r2_realisationst_upper_uq=zeros(2,length(t),nptss)

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
                predict_r2_realisations_upper_lq[:,:,i]= [quantile(data_dist, THrealisations_prediction/2) for data_dist in loop_data_dists];
                predict_r2_realisations_upper_uq[:,:,i]= [quantile(data_dist, 1-THrealisations_prediction/2) for data_dist in loop_data_dists];

                loop_data_distst=[Normal(mi,Sd) for mi in model(t,[nr2range_upper[1,i],r2range_upper[i]])]; # normal distribution about mean
                predict_r2_realisationst_upper_lq[:,:,i]= [quantile(data_dist, THrealisations_prediction/2) for data_dist in loop_data_distst];
                predict_r2_realisationst_upper_uq[:,:,i]= [quantile(data_dist, 1-THrealisations_prediction/2) for data_dist in loop_data_distst];


                if fo > fmle
                    # println("new MLE r2 upper")
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
                predict_r2_realisations_lower_lq[:,:,i]= [quantile(data_dist, THrealisations_prediction/2) for data_dist in loop_data_dists];
                predict_r2_realisations_lower_uq[:,:,i]= [quantile(data_dist, 1-THrealisations_prediction/2) for data_dist in loop_data_dists];
                
                loop_data_distst=[Normal(mi,Sd) for mi in model(t,[nr2range_lower[1,i],r2range_lower[i]])]; # normal distribution about mean
                predict_r2_realisationst_lower_lq[:,:,i]= [quantile(data_dist, THrealisations_prediction/2) for data_dist in loop_data_distst];
                predict_r2_realisationst_lower_uq[:,:,i]= [quantile(data_dist, 1-THrealisations_prediction/2) for data_dist in loop_data_distst];
                

                if fo > fmle
                    # println("new MLE r2 lower")
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
                if (llr2_lower[i].-maximum(llr2)) >= THrealisations_profile
                    for j in 1:length(t_smooth)
                        max_realisations_r2[1,j]=max(predict_r2_realisations_lower_uq[1,j,i],max_realisations_r2[1,j])
                        min_realisations_r2[1,j]=min(predict_r2_realisations_lower_lq[1,j,i],min_realisations_r2[1,j])
                        max_realisations_r2[2,j]=max(predict_r2_realisations_lower_uq[2,j,i],max_realisations_r2[2,j])
                        min_realisations_r2[2,j]=min(predict_r2_realisations_lower_lq[2,j,i],min_realisations_r2[2,j])
                    end
                end
            end
            for i in 1:(nptss)
                if (llr2_upper[i].-maximum(llr2)) >= THrealisations_profile
                    for j in 1:length(t_smooth)
                        max_realisations_r2[1,j]=max(predict_r2_realisations_upper_uq[1,j,i],max_realisations_r2[1,j])
                        min_realisations_r2[1,j]=min(predict_r2_realisations_upper_lq[1,j,i],min_realisations_r2[1,j]) 
                        max_realisations_r2[2,j]=max(predict_r2_realisations_upper_uq[2,j,i],max_realisations_r2[2,j])
                        min_realisations_r2[2,j]=min(predict_r2_realisations_upper_lq[2,j,i],min_realisations_r2[2,j])
                    end
                end
            end
               # combine the confidence sets for realisations
               max_realisationst_r2=zeros(2,length(t))
               min_realisationst_r2=1000*ones(2,length(t))
               for i in 1:(nptss)
                   if (llr2_lower[i].-maximum(llr2)) >= THrealisations_profile
                       for j in 1:length(t)
                           max_realisationst_r2[1,j]=max(predict_r2_realisationst_lower_uq[1,j,i],max_realisationst_r2[1,j])
                           min_realisationst_r2[1,j]=min(predict_r2_realisationst_lower_lq[1,j,i],min_realisationst_r2[1,j])
                           max_realisationst_r2[2,j]=max(predict_r2_realisationst_lower_uq[2,j,i],max_realisationst_r2[2,j])
                           min_realisationst_r2[2,j]=min(predict_r2_realisationst_lower_lq[2,j,i],min_realisationst_r2[2,j])
                       end
                   end
               end
               for i in 1:(nptss)
                   if (llr2_upper[i].-maximum(llr2)) >= THrealisations_profile
                       for j in 1:length(t)
                           max_realisationst_r2[1,j]=max(predict_r2_realisationst_upper_uq[1,j,i],max_realisationst_r2[1,j])
                           min_realisationst_r2[1,j]=min(predict_r2_realisationst_upper_lq[1,j,i],min_realisationst_r2[1,j]) 
                           max_realisationst_r2[2,j]=max(predict_r2_realisationst_upper_uq[2,j,i],max_realisationst_r2[2,j])
                           min_realisationst_r2[2,j]=min(predict_r2_realisationst_upper_lq[2,j,i],min_realisationst_r2[2,j])
                       end
                   end
               end
            


            ##############################################################################################################
            combined_plot_linewidth = 2;

            # interpolate for smoother profile likelihoods
            interp_nptss= 1001;
            
            # r1
            interp_points_r1range =  LinRange(r1min,r1max,interp_nptss)
            interp_r1 = LinearInterpolation(r1range,nllr1)
            interp_nllr1 = interp_r1(interp_points_r1range)
            
            # r2
            interp_points_r2range =  LinRange(r2min,r2max,interp_nptss)
            interp_r2 = LinearInterpolation(r2range,nllr2)
            interp_nllr2 = interp_r2(interp_points_r2range)
                        
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
            if (r1 >= lb_CI_r1 ).*(r1 <= ub_CI_r1)
                r1_withinbounds[seednum] = 1
            end
            
            # r2
            (lb_CI_r2,ub_CI_r2) = fun_interpCI(r2mle,interp_points_r2range,interp_nllr2,TH)
            if (r2 >= lb_CI_r2 ).*(r2 <= ub_CI_r2)
                r2_withinbounds[seednum] =1
            end       
    

    #########################################################################################
            # Parameter-wise profile predictions
    
            # overall - plot model simulated at MLE, scatter data, and prediction interval.
            max_overall=zeros(2,length(t_smooth))
            min_overall=1000*ones(2,length(t_smooth))
            for j in 1:length(t_smooth)
                max_overall[1,j]=max(max_r1[1,j],max_r2[1,j])
                max_overall[2,j]=max(max_r1[2,j],max_r2[2,j])
                min_overall[1,j]=min(min_r1[1,j],min_r2[1,j])
                min_overall[2,j]=min(min_r1[2,j],min_r2[2,j])
            end
            
            ytrue_smooth =  model(t_smooth,[r1,r2,C10,C20]); 

            # min_r1
            # max_r1

            # sum(ytrue_smooth .>= min_r1) == length(t_smooth)

            # is the true model solution within the model solution bounds from r1
            if (sum(ytrue_smooth .>= min_r1) == 2*length(t_smooth)) .* (sum(ytrue_smooth .<= max_r1) == 2*length(t_smooth))
                r1modelsoln_withinbounds[seednum] = 1
            end


            # is the true model solution within the model solution bounds from r2
            if (sum(ytrue_smooth .>= min_r2) == 2*length(t_smooth)) .* (sum(ytrue_smooth .<= max_r2) == 2*length(t_smooth))
                r2modelsoln_withinbounds[seednum] = 1
            end


              # is the true model solution within the model solution bounds from r1
              if (sum(ytrue_smooth .>= min_overall) == 2*length(t_smooth)) .* (sum(ytrue_smooth .<= max_overall) == 2*length(t_smooth))
                modelsoln_withinbounds[seednum] = 1
            end


            # count of number of points of the true solution that are within the bounds of the confidence set for the model solution

            r1modelsoln_countwithinbounds[seednum] = sum((ytrue_smooth[:,2:end] .>= min_r1[:,2:end]) .* (ytrue_smooth[:,2:end] .<= max_r1[:,2:end]) )
            r2modelsoln_countwithinbounds[seednum] = sum((ytrue_smooth[:,2:end] .>= min_r2[:,2:end]) .* (ytrue_smooth[:,2:end] .<= max_r2[:,2:end]) )
            modelsoln_countwithinbounds[seednum] = sum((ytrue_smooth[:,2:end] .>= min_overall[:,2:end]) .* (ytrue_smooth[:,2:end] .<= max_overall[:,2:end]) )
            

        ## time
        for i in 1:length(t_smooth)
            # println(i)
            # model solution - r1
            if (ytrue_smooth[1,i] .>= min_r1[1,i]) .* (ytrue_smooth[1,i] .<= max_r1[1,i])
                r1modelsoln_withinbounds_time[seednum,1,i] = 1
            end
            if (ytrue_smooth[2,i] .>= min_r1[2,i]) .* (ytrue_smooth[2,i] .<= max_r1[2,i])
                r1modelsoln_withinbounds_time[seednum,2,i] = 1
            end
            # model solution - r2
            if (ytrue_smooth[1,i] .>= min_r2[1,i]) .* (ytrue_smooth[1,i] .<= max_r2[1,i])
                r2modelsoln_withinbounds_time[seednum,1,i] = 1
            end
            if (ytrue_smooth[2,i] .>= min_r2[2,i]) .* (ytrue_smooth[2,i] .<= max_r2[2,i])
                r2modelsoln_withinbounds_time[seednum,2,i] = 1
            end

            # model solution - union
            if (ytrue_smooth[1,i] .>= min_overall[1,i]) .* (ytrue_smooth[1,i] .<= max_overall[1,i])
                modelsoln_withinbounds_time[seednum,1,i] = 1
            end
            if (ytrue_smooth[2,i] .>= min_overall[2,i]) .* (ytrue_smooth[2,i] .<= max_overall[2,i])
                modelsoln_withinbounds_time[seednum,2,i] = 1
            end
        end

           



            ##############################
            # # plot difference of true model solutions vs confidence sets for models solutions

          if seednum == 1

            #### r1
            modelsolnconfidenceset_r1_truesoln=plot(t_smooth,data0_smooth_MLE[1,:],w=0,c=colour1,ribbon=(data0_smooth_MLE[1,:].-min_r1[1,:],max_r1[1,:].-data0_smooth_MLE[1,:]),fillalpha=.2,xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,legend=false)
            modelsolnconfidenceset_r1_truesoln=plot!(t_smooth,data0_smooth_MLE[2,:],w=0,c=colour2,ribbon=(data0_smooth_MLE[2,:].-min_r1[2,:],max_r1[2,:].-data0_smooth_MLE[2,:]),fillalpha=.2,xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt)
            modelsolnconfidenceset_r1_truesoln=plot!(t_smooth,ytrue_smooth[1,:],lw=3,linecolor=:black) # c1 true
            modelsolnconfidenceset_r1_truesoln=plot!(t_smooth,ytrue_smooth[2,:],lw=3,linecolor=:black) # c1 true
            savefig(modelsolnconfidenceset_r1_truesoln,filepath_save[1] * "Fig_" * "modelsolnconfidenceset_r1_truesoln_i"  * string(seednum) *  ".pdf")

            #### r2
            modelsolnconfidenceset_r2_truesoln=plot(t_smooth,data0_smooth_MLE[1,:],w=0,c=colour1,ribbon=(data0_smooth_MLE[1,:].-min_r2[1,:],max_r2[1,:].-data0_smooth_MLE[1,:]),fillalpha=.2,xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,legend=false)
            modelsolnconfidenceset_r2_truesoln=plot!(t_smooth,data0_smooth_MLE[2,:],w=0,c=colour2,ribbon=(data0_smooth_MLE[2,:].-min_r2[2,:],max_r2[2,:].-data0_smooth_MLE[2,:]),fillalpha=.2,xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt)
            modelsolnconfidenceset_r2_truesoln=plot!(t_smooth,ytrue_smooth[1,:],lw=3,linecolor=:black) # c1 true
            modelsolnconfidenceset_r2_truesoln=plot!(t_smooth,ytrue_smooth[2,:],lw=3,linecolor=:black) # c1 true
            savefig(modelsolnconfidenceset_r2_truesoln,filepath_save[1] * "Fig_" * "modelsolnconfidenceset_r2_truesoln_i"  * string(seednum) *  ".pdf")

            #### union 
            
            modelsolnconfidenceset_union_truesoln=plot(t_smooth,data0_smooth_MLE[1,:],w=0,c=colour1,ribbon=(data0_smooth_MLE[1,:].-min_overall[1,:],max_overall[1,:].-data0_smooth_MLE[1,:]),fillalpha=.2,xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,legend=false)
            modelsolnconfidenceset_union_truesoln=plot!(t_smooth,data0_smooth_MLE[2,:],w=0,c=colour2,ribbon=(data0_smooth_MLE[2,:].-min_overall[2,:],max_overall[2,:].-data0_smooth_MLE[2,:]),fillalpha=.2,xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt)
            modelsolnconfidenceset_union_truesoln=plot!(t_smooth,ytrue_smooth[1,:],lw=3,linecolor=:black) # c1 true
            modelsolnconfidenceset_union_truesoln=plot!(t_smooth,ytrue_smooth[2,:],lw=3,linecolor=:black) # c1 true
            savefig(modelsolnconfidenceset_union_truesoln,filepath_save[1] * "Fig_" * "modelsolnconfidenceset_union_truesoln_i"  * string(seednum) *  ".pdf")


          end


            ##############################
            # # plot difference of confidence sets for models solutions ribbon to zero
           
           if seednum == 1
            ydiffmin=-4.2
            ydiffmax=4.2


            #### r1
            modelsolnconfidenceset_r1_truesoln_diff_c1=plot(t_smooth,0 .*data0_smooth_MLE[1,:],w=0,c=colour1,ribbon=(data0_smooth_MLE[1,:].-min_r1[1,:],max_r1[1,:].-data0_smooth_MLE[1,:]),ylims=(ydiffmin,ydiffmax),legend=false,fillalpha=.2,xlab=L"t",ylab=L"C_{y,0.95}^{r_{1}} - y(\hat{\theta})",lw=0,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour1)
            modelsolnconfidenceset_r1_truesoln_diff_c1=plot!(t_smooth,ytrue_smooth[1,:] .- data0_smooth_MLE[1,:],lw=3,linecolor=:black)
            savefig(modelsolnconfidenceset_r1_truesoln_diff_c1,filepath_save[1] * "Fig_" * "modelsolnconfidenceset_r1_truesoln_diff_c1_i"  * string(seednum) *  ".pdf")
          
            modelsolnconfidenceset_r1_truesoln_diff_c2=plot(t_smooth,0 .*data0_smooth_MLE[2,:],w=0,c=colour2,ribbon=(data0_smooth_MLE[2,:].-min_r1[2,:],max_r1[2,:].-data0_smooth_MLE[2,:]),ylims=(ydiffmin,ydiffmax),fillalpha=.2,xlab=L"t",ylab=L"C_{y,0.95}^{r_{1}} - y(\hat{\theta})",lw=0,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2,legend=false)
            modelsolnconfidenceset_r1_truesoln_diff_c2=plot!(t_smooth,ytrue_smooth[2,:] .- data0_smooth_MLE[2,:],lw=3,linecolor=:black)
            savefig(modelsolnconfidenceset_r1_truesoln_diff_c2,filepath_save[1] * "Fig_" * "modelsolnconfidenceset_r1_truesoln_diff_c2_i" * string(seednum) * ".pdf")
          

            #### r2 
            modelsolnconfidenceset_r2_truesoln_diff_c1=plot(t_smooth,0 .*data0_smooth_MLE[1,:],w=0,c=colour1,ribbon=(data0_smooth_MLE[1,:].-min_r2[1,:],max_r2[1,:].-data0_smooth_MLE[1,:]),ylims=(ydiffmin,ydiffmax),legend=false,fillalpha=.2,xlab=L"t",ylab=L"C_{y,0.95}^{r_{2}} - y(\hat{\theta})",lw=0,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour1)
            modelsolnconfidenceset_r2_truesoln_diff_c1=plot!(t_smooth,ytrue_smooth[1,:] .- data0_smooth_MLE[1,:],lw=3, linecolor=:black)
            savefig(modelsolnconfidenceset_r2_truesoln_diff_c1,filepath_save[1] * "Fig_" * "modelsolnconfidenceset_r2_truesoln_diff_c1_i"  * string(seednum) *  ".pdf")
          
            modelsolnconfidenceset_r2_truesoln_diff_c2=plot(t_smooth,0 .*data0_smooth_MLE[2,:],w=0,c=colour2,ribbon=(data0_smooth_MLE[2,:].-min_r2[2,:],max_r2[2,:].-data0_smooth_MLE[2,:]),ylims=(ydiffmin,ydiffmax),fillalpha=.2,xlab=L"t",ylab=L"C_{y,0.95}^{r_{2}} - y(\hat{\theta})",lw=0,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2,legend=false)
            modelsolnconfidenceset_r2_truesoln_diff_c2=plot!(t_smooth,ytrue_smooth[2,:] .- data0_smooth_MLE[2,:],lw=3, linecolor =:black)
            savefig(modelsolnconfidenceset_r2_truesoln_diff_c2,filepath_save[1] * "Fig_" * "modelsolnconfidenceset_r2_truesoln_diff_c2_i" * string(seednum) * ".pdf")
          

            #### union
            modelsolnconfidenceset_union_truesoln_diff_c1=plot(t_smooth,0 .*data0_smooth_MLE[1,:],w=0,c=colour1,ribbon=(data0_smooth_MLE[1,:].-min_overall[1,:],max_overall[1,:].-data0_smooth_MLE[1,:]),ylims=(ydiffmin,ydiffmax),legend=false,fillalpha=.2,xlab=L"t",ylab=L"C_{y,0.95} - y(\hat{\theta})",lw=0,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour1)
            modelsolnconfidenceset_union_truesoln_diff_c1=plot!(t_smooth,ytrue_smooth[1,:] .- data0_smooth_MLE[1,:],lw=3, linecolor=:black)
            savefig(modelsolnconfidenceset_union_truesoln_diff_c1,filepath_save[1] * "Fig_" * "modelsolnconfidenceset_union_truesoln_diff_c1_i"  * string(seednum) *  ".pdf")
          
            modelsolnconfidenceset_union_truesoln_diff_c2=plot(t_smooth,0 .*data0_smooth_MLE[2,:],w=0,c=colour2,ribbon=(data0_smooth_MLE[2,:].-min_overall[2,:],max_overall[2,:].-data0_smooth_MLE[2,:]),ylims=(ydiffmin,ydiffmax),fillalpha=.2,xlab=L"t",ylab=L"C_{y,0.95} - y(\hat{\theta})",lw=0,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2,legend=false)
            modelsolnconfidenceset_union_truesoln_diff_c2=plot!(t_smooth,ytrue_smooth[2,:] .- data0_smooth_MLE[2,:],lw=3, linecolor=:black)
            savefig(modelsolnconfidenceset_union_truesoln_diff_c2,filepath_save[1] * "Fig_" * "modelsolnconfidenceset_union_truesoln_diff_c2_i" * string(seednum) * ".pdf")
          

            
      
           end
            



                # overall - plot model simulated at MLE, scatter data, and prediction interval (union of confidence sets for realisations)
                max_realisations_overall=zeros(2,length(t_smooth))
                min_realisations_overall=1000*ones(2,length(t_smooth))
                for j in 1:length(t_smooth)
                    max_realisations_overall[1,j]=max(max_realisations_r1[1,j],max_realisations_r2[1,j])
                    max_realisations_overall[2,j]=max(max_realisations_r1[2,j],max_realisations_r2[2,j])
                    min_realisations_overall[1,j]=min(min_realisations_r1[1,j],min_realisations_r2[1,j])
                    min_realisations_overall[2,j]=min(min_realisations_r1[2,j],min_realisations_r2[2,j])
                end

                max_realisationst_overall=zeros(2,length(t))
                min_realisationst_overall=1000*ones(2,length(t))
                for j in 1:length(t)
                    max_realisationst_overall[1,j]=max(max_realisationst_r1[1,j],max_realisationst_r2[1,j])
                    max_realisationst_overall[2,j]=max(max_realisationst_r1[2,j],max_realisationst_r2[2,j])
                    min_realisationst_overall[1,j]=min(min_realisationst_r1[1,j],min_realisationst_r2[1,j])
                    min_realisationst_overall[2,j]=min(min_realisationst_r1[2,j],min_realisationst_r2[2,j])
                end



                      
        for i in 1:length(t)
            
            # data realisations - r1
            if (testdata[1,i] .>= min_realisationst_r1[1,i]) .* (testdata[1,i] .<= max_realisationst_r1[1,i])
                r1realisationst_withinbounds_time[seednum,1,i] = 1
            end
            if (testdata[2,i] .>= min_realisationst_r1[2,i]) .* (testdata[2,i] .<= max_realisationst_r1[2,i])
                r1realisationst_withinbounds_time[seednum,2,i] = 1
            end

            # data realisations - r2
            if (testdata[1,i] .>= min_realisationst_r2[1,i]) .* (testdata[1,i] .<= max_realisationst_r2[1,i])
                r2realisationst_withinbounds_time[seednum,1,i] = 1
            end
            if (testdata[2,i] .>= min_realisationst_r2[2,i]) .* (testdata[2,i] .<= max_realisationst_r2[2,i])
                r2realisationst_withinbounds_time[seednum,2,i] = 1
            end

            # data realisations - union
            if (testdata[1,i] .>= min_realisationst_overall[1,i]) .* (testdata[1,i] .<= max_realisationst_overall[1,i])
                realisationst_withinbounds_time[seednum,1,i] = 1
            end
            if (testdata[2,i] .>= min_realisationst_overall[2,i]) .* (testdata[2,i] .<= max_realisationst_overall[2,i])
                realisationst_withinbounds_time[seednum,2,i] = 1
            end
        end


                # is the true model solution within the model solution bounds from r1
                if (sum(testdata .>= min_realisationst_r1) == 2*length(t)) .* (sum(testdata .<= max_realisationst_r1) == 2*length(t))
                    r1realisations_withinbounds[seednum] = 1
                end


                # is the true model solution within the model solution bounds from r2
                if (sum(testdata .>= min_realisationst_r2) == 2*length(t)) .* (sum(testdata .<= max_realisationst_r2) == 2*length(t))
                    r2realisations_withinbounds[seednum] = 1
                end


                # is the true model solution within the model solution bounds from r1
                if (sum(testdata .>= min_realisationst_overall) == 2*length(t)) .* (sum(testdata .<= max_realisationst_overall) == 2*length(t))
                    realisations_withinbounds[seednum] = 1
                end


                    ## Plot confidence sets for realisations
                if seednum == 1

                    #### r1
                    modelrealconfidenceset_r1_truesoln=plot(t_smooth,data0_smooth_MLE[1,:],w=0,c=colour1,ribbon=(data0_smooth_MLE[1,:].-min_realisations_r1[1,:],max_realisations_r1[1,:].-data0_smooth_MLE[1,:]),fillalpha=.2,xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,legend=false,ylim=(ymin,ymax))
                    modelrealconfidenceset_r1_truesoln=plot!(t_smooth,data0_smooth_MLE[2,:],w=0,c=colour2,ribbon=(data0_smooth_MLE[2,:].-min_realisations_r1[2,:],max_realisations_r1[2,:].-data0_smooth_MLE[2,:]),fillalpha=.2,xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt)
                    modelrealconfidenceset_r1_truesoln=plot!(t_smooth,ytrue_smooth[1,:],lw=3,linecolor=:black) # c1 true
                    modelrealconfidenceset_r1_truesoln=plot!(t_smooth,ytrue_smooth[2,:],lw=3,linecolor=:black) # c1 true
                    modelrealconfidenceset_r1_truesoln=scatter!(t,testdata[1,:],markersize = 8,markercolor=:orange,msw=0)
                    modelrealconfidenceset_r1_truesoln=scatter!(t,testdata[2,:],markersize = 8,markercolor=:blue,msw=0)

                    savefig(modelrealconfidenceset_r1_truesoln,filepath_save[1] * "Fig_" * "modelrealconfidenceset_r1_truesoln_i"  * string(seednum) *  ".pdf")
        
                    #### r2
                    modelrealconfidenceset_r2_truesoln=plot(t_smooth,data0_smooth_MLE[1,:],w=0,c=colour1,ribbon=(data0_smooth_MLE[1,:].-min_realisations_r2[1,:],max_realisations_r2[1,:].-data0_smooth_MLE[1,:]),fillalpha=.2,xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,legend=false,ylim=(ymin,ymax))
                    modelrealconfidenceset_r2_truesoln=plot!(t_smooth,data0_smooth_MLE[2,:],w=0,c=colour2,ribbon=(data0_smooth_MLE[2,:].-min_realisations_r2[2,:],max_realisations_r2[2,:].-data0_smooth_MLE[2,:]),fillalpha=.2,xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt)
                    modelrealconfidenceset_r2_truesoln=plot!(t_smooth,ytrue_smooth[1,:],lw=3,linecolor=:black) # c1 true
                    modelrealconfidenceset_r2_truesoln=plot!(t_smooth,ytrue_smooth[2,:],lw=3,linecolor=:black) # c1 true
                    modelrealconfidenceset_r2_truesoln=scatter!(t,testdata[1,:],markersize = 8,markercolor=:orange,msw=0)
                    modelrealconfidenceset_r2_truesoln=scatter!(t,testdata[2,:],markersize = 8,markercolor=:blue,msw=0)

                    savefig(modelrealconfidenceset_r2_truesoln,filepath_save[1] * "Fig_" * "modelrealconfidenceset_r2_truesoln_i"  * string(seednum) *  ".pdf")
        
                    #### union 
                    
                    modelrealconfidenceset_union_truesoln=plot(t_smooth,data0_smooth_MLE[1,:],w=0,c=colour1,ribbon=(data0_smooth_MLE[1,:].-min_realisations_overall[1,:],max_realisations_overall[1,:].-data0_smooth_MLE[1,:]),fillalpha=.2,xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,legend=false,ylim=(ymin,ymax))
                    modelrealconfidenceset_union_truesoln=plot!(t_smooth,data0_smooth_MLE[2,:],w=0,c=colour2,ribbon=(data0_smooth_MLE[2,:].-min_realisations_overall[2,:],max_realisations_overall[2,:].-data0_smooth_MLE[2,:]),fillalpha=.2,xlab=L"t",ylab=L"c_{1}(t), c_{2}(t)",titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt)
                    modelrealconfidenceset_union_truesoln=plot!(t_smooth,ytrue_smooth[1,:],lw=3,linecolor=:black) # c1 true
                    modelrealconfidenceset_union_truesoln=plot!(t_smooth,ytrue_smooth[2,:],lw=3,linecolor=:black) # c1 true
                    modelrealconfidenceset_union_truesoln=scatter!(t,testdata[1,:],markersize = 8,markercolor=:orange,msw=0)
                    modelrealconfidenceset_union_truesoln=scatter!(t,testdata[2,:],markersize = 8,markercolor=:blue,msw=0)

                    savefig(modelrealconfidenceset_union_truesoln,filepath_save[1] * "Fig_" * "modelrealconfidenceset_union_truesoln_i"  * string(seednum) *  ".pdf")
        
        
                  end

              
            ##############################
            # # plot difference of confidence sets for models solutions ribbon to zero
           
           if seednum == 1
            ydiffmin=-20
            ydiffmax=20


            #### r1
            modelrealconfidenceset_r1_truesoln_diff_c1=plot(t_smooth,0 .*data0_smooth_MLE[1,:],w=0,c=colour1,ribbon=(data0_smooth_MLE[1,:].-min_realisations_r1[1,:],max_realisations_r1[1,:].-data0_smooth_MLE[1,:]),ylims=(ydiffmin,ydiffmax),legend=false,fillalpha=.2,xlab=L"t",ylab=L"C_{z_{i},0.95}^{r_{1}} - y(\hat{\theta})",lw=0,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour1)
            modelrealconfidenceset_r1_truesoln_diff_c1=plot!(t_smooth,ytrue_smooth[1,:] .- data0_smooth_MLE[1,:],lw=3,linecolor=:black)
            modelrealconfidenceset_r1_truesoln_diff_c1=scatter!(t,testdata[1,:]-data0_MLE[1,:],legend=false,markersize = 8,markercolor=:orange,msw=0)
            savefig(modelrealconfidenceset_r1_truesoln_diff_c1,filepath_save[1] * "Fig_" * "modelrealconfidenceset_r1_truesoln_diff_c1_i"  * string(seednum) *  ".pdf")
          
            modelrealconfidenceset_r1_truesoln_diff_c2=plot(t_smooth,0 .*data0_smooth_MLE[2,:],w=0,c=colour2,ribbon=(data0_smooth_MLE[2,:].-min_realisations_r1[2,:],max_realisations_r1[2,:].-data0_smooth_MLE[2,:]),ylims=(ydiffmin,ydiffmax),fillalpha=.2,xlab=L"t",ylab=L"C_{z_{i},0.95}^{r_{1}} - y(\hat{\theta})",lw=0,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2,legend=false)
            modelrealconfidenceset_r1_truesoln_diff_c2=plot!(t_smooth,ytrue_smooth[2,:] .- data0_smooth_MLE[2,:],lw=3,linecolor=:black)
            modelrealconfidenceset_r1_truesoln_diff_c2=scatter!(t,testdata[2,:]-data0_MLE[2,:],legend=false,markersize = 8,markercolor=:blue,msw=0)
            savefig(modelrealconfidenceset_r1_truesoln_diff_c2,filepath_save[1] * "Fig_" * "modelrealconfidenceset_r1_truesoln_diff_c2_i" * string(seednum) * ".pdf")
          

            #### r2 
            modelrealconfidenceset_r2_truesoln_diff_c1=plot(t_smooth,0 .*data0_smooth_MLE[1,:],w=0,c=colour1,ribbon=(data0_smooth_MLE[1,:].-min_realisations_r2[1,:],max_realisations_r2[1,:].-data0_smooth_MLE[1,:]),ylims=(ydiffmin,ydiffmax),legend=false,fillalpha=.2,xlab=L"t",ylab=L"C_{z_{i},0.95}^{r_{2}} - y(\hat{\theta})",lw=0,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour1)
            modelrealconfidenceset_r2_truesoln_diff_c1=plot!(t_smooth,ytrue_smooth[1,:] .- data0_smooth_MLE[1,:],lw=3, linecolor=:black)
            modelrealconfidenceset_r2_truesoln_diff_c1=scatter!(t,testdata[1,:]-data0_MLE[1,:],legend=false,markersize = 8,markercolor=:orange,msw=0)
            savefig(modelrealconfidenceset_r2_truesoln_diff_c1,filepath_save[1] * "Fig_" * "modelrealconfidenceset_r2_truesoln_diff_c1_i"  * string(seednum) *  ".pdf")
          
            modelrealconfidenceset_r2_truesoln_diff_c2=plot(t_smooth,0 .*data0_smooth_MLE[2,:],w=0,c=colour2,ribbon=(data0_smooth_MLE[2,:].-min_realisations_r2[2,:],max_realisations_r2[2,:].-data0_smooth_MLE[2,:]),ylims=(ydiffmin,ydiffmax),fillalpha=.2,xlab=L"t",ylab=L"C_{z_{i},0.95}^{r_{2}} - y(\hat{\theta})",lw=0,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2,legend=false)
            modelrealconfidenceset_r2_truesoln_diff_c2=plot!(t_smooth,ytrue_smooth[2,:] .- data0_smooth_MLE[2,:],lw=3, linecolor =:black)
            modelrealconfidenceset_r2_truesoln_diff_c2=scatter!(t,testdata[2,:]-data0_MLE[2,:],legend=false,markersize = 8,markercolor=:blue,msw=0)
            savefig(modelrealconfidenceset_r2_truesoln_diff_c2,filepath_save[1] * "Fig_" * "modelrealconfidenceset_r2_truesoln_diff_c2_i" * string(seednum) * ".pdf")
          

            #### union
            modelrealconfidenceset_union_truesoln_diff_c1=plot(t_smooth,0 .*data0_smooth_MLE[1,:],w=0,c=colour1,ribbon=(data0_smooth_MLE[1,:].-min_realisations_overall[1,:],max_realisations_overall[1,:].-data0_smooth_MLE[1,:]),ylims=(ydiffmin,ydiffmax),legend=false,fillalpha=.2,xlab=L"t",ylab=L"C_{z_{i},0.95} - y(\hat{\theta})",lw=0,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour1)
            modelrealconfidenceset_union_truesoln_diff_c1=plot!(t_smooth,ytrue_smooth[1,:] .- data0_smooth_MLE[1,:],lw=3, linecolor=:black)
            modelrealconfidenceset_union_truesoln_diff_c1=scatter!(t,testdata[1,:]-data0_MLE[1,:],legend=false,markersize = 8,markercolor=:orange,msw=0)
            savefig(modelrealconfidenceset_union_truesoln_diff_c1,filepath_save[1] * "Fig_" * "modelrealconfidenceset_union_truesoln_diff_c1_i"  * string(seednum) *  ".pdf")
          
            modelrealconfidenceset_union_truesoln_diff_c2=plot(t_smooth,0 .*data0_smooth_MLE[2,:],w=0,c=colour2,ribbon=(data0_smooth_MLE[2,:].-min_realisations_overall[2,:],max_realisations_overall[2,:].-data0_smooth_MLE[2,:]),ylims=(ydiffmin,ydiffmax),fillalpha=.2,xlab=L"t",ylab=L"C_{z_{i},0.95} - y(\hat{\theta})",lw=0,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2,legend=false)
            modelrealconfidenceset_union_truesoln_diff_c2=plot!(t_smooth,ytrue_smooth[2,:] .- data0_smooth_MLE[2,:],lw=3, linecolor=:black)
            modelrealconfidenceset_union_truesoln_diff_c2=scatter!(t,testdata[2,:]-data0_MLE[2,:],legend=false,markersize = 8,markercolor=:blue,msw=0)
            savefig(modelrealconfidenceset_union_truesoln_diff_c2,filepath_save[1] * "Fig_" * "modelrealconfidenceset_union_truesoln_diff_c2_i" * string(seednum) * ".pdf")
          
           end


        end

        println("End: ")
        println(now())



            r1coverage = sum(r1_withinbounds)/totalseeds
            r2coverage = sum(r2_withinbounds)/totalseeds
            
            r1modelcoverage = sum(r1modelsoln_withinbounds)/totalseeds
            r2modelcoverage = sum(r2modelsoln_withinbounds)/totalseeds
            unionmodelcoverage = sum(modelsoln_withinbounds)/totalseeds
            
            r1realisationscoverage = sum(r1realisations_withinbounds)/totalseeds
            r2realisationscoverage = sum(r2realisations_withinbounds)/totalseeds
            unionrealisationscoverage = sum(realisations_withinbounds)/totalseeds

            realisationsMLEcoverage= sum(realisations_MLE_withinbounds)/totalseeds
            
            if @isdefined(df_coverage) == 0
                println("not defined")
                global df_coverage = DataFrame( r1coverage=r1coverage, r2coverage=r2coverage, r1modelcoverage = r1modelcoverage, r2modelcoverage= r2modelcoverage, unionmodelcoverage=unionmodelcoverage,r1realisationscoverage=r1realisationscoverage,r2realisationscoverage=r2realisationscoverage, unionrealisationscoverage=unionrealisationscoverage)
            else 
                println("DEFINED")
                global df_coverage = DataFrame(  r1coverage=r1coverage, r2coverage=r2coverage,r1modelcoverage = r1modelcoverage, r2modelcoverage= r2modelcoverage, unionmodelcoverage=unionmodelcoverage, r1realisationscoverage=r1realisationscoverage,r2realisationscoverage=r2realisationscoverage, unionrealisationscoverage=unionrealisationscoverage)
            end
            
            CSV.write(filepath_save[1] * "df_coverage.csv", df_coverage)
            
        
### Plots


# Model solution  - r1
coverage_m_r1_c1 = plot(t_smooth[2:end], sum(r1modelsoln_withinbounds_time[:,1,2:end],dims=1)' ./totalseeds, ylim=(0,1.1),lw=3, titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour1,legend=false,xlab=L"t",ylab=L"C_{y,0.95}^{r_{1}}",xticks=([0,1,2]),yticks=([0,0.5,1.0]))
coverage_m_r1_c1 = hline!([1.00],lw=2,linecolor=:black,linestyle=:dash)
coverage_m_r1_c1 = hline!([0.95],lw=2,linecolor=:black,linestyle=:dot)
display(coverage_m_r1_c1)
savefig(coverage_m_r1_c1,filepath_save[1] * "Fig_" * "coverage_m_r1_c1"  * ".pdf")

coverage_m_r1_c2 = plot(t_smooth[2:end], sum(r1modelsoln_withinbounds_time[:,2,2:end],dims=1)' ./totalseeds, ylim=(0,1.1),lw=3, titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2,legend=false,xlab=L"t",ylab=L"C_{y,0.95}^{r_{1}}",xticks=([0,1,2]),yticks=([0,0.5,1.0]))
coverage_m_r1_c2 = hline!([1.00],lw=2,linecolor=:black,linestyle=:dash)
coverage_m_r1_c2 = hline!([0.95],lw=2,linecolor=:black,linestyle=:dot)
display(coverage_m_r1_c2)
savefig(coverage_m_r1_c2,filepath_save[1] * "Fig_" * "coverage_m_r1_c2"  * ".pdf")

# Model solution  - r2
coverage_m_r2_c1 = plot(t_smooth[2:end], sum(r2modelsoln_withinbounds_time[:,1,2:end],dims=1)' ./totalseeds, ylim=(0,1.1),lw=3, titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour1,legend=false,xlab=L"t",ylab=L"C_{y,0.95}^{r_{2}}",xticks=([0,1,2]),yticks=([0,0.5,1.0]))
coverage_m_r2_c1 = hline!([1.00],lw=2,linecolor=:black,linestyle=:dash)
coverage_m_r2_c1 = hline!([0.95],lw=2,linecolor=:black,linestyle=:dot)
display(coverage_m_r2_c1)
savefig(coverage_m_r2_c1,filepath_save[1] * "Fig_" * "coverage_m_r2_c1"  * ".pdf")

coverage_m_r2_c2 = plot(t_smooth[2:end], sum(r2modelsoln_withinbounds_time[:,2,2:end],dims=1)' ./totalseeds, ylim=(0,1.1),lw=3, titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2,legend=false,xlab=L"t",ylab=L"C_{y,0.95}^{r_{2}}",xticks=([0,1,2]),yticks=([0,0.5,1.0]))
coverage_m_r2_c2 = hline!([1.00],lw=2,linecolor=:black,linestyle=:dash)
coverage_m_r2_c2 = hline!([0.95],lw=2,linecolor=:black,linestyle=:dot)
display(coverage_m_r2_c2)
savefig(coverage_m_r2_c2,filepath_save[1] * "Fig_" * "coverage_m_r2_c2"  * ".pdf")

# Model solution  - union
coverage_m_union_c1  = plot(t_smooth[2:end], sum(modelsoln_withinbounds_time[:,1,2:end],dims=1)' ./totalseeds, ylim=(0,1.1),lw=3, titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour1,legend=false,xlab=L"t",ylab=L"C_{y,0.95}",xticks=([0,1,2]),yticks=([0,0.5,1.0]))
coverage_m_union_c1 = hline!([1.00],lw=2,linecolor=:black,linestyle=:dash)
coverage_m_union_c1 = hline!([0.95],lw=2,linecolor=:black,linestyle=:dot)
display(coverage_m_union_c1)
savefig(coverage_m_union_c1,filepath_save[1] * "Fig_" * "coverage_m_union_c1"  * ".pdf")

coverage_m_union_c2 = plot(t_smooth[2:end], sum(modelsoln_withinbounds_time[:,2,2:end],dims=1)' ./totalseeds, ylim=(0,1.1),lw=3, titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2,legend=false,xlab=L"t",ylab=L"C_{y,0.95}",xticks=([0,1,2]),yticks=([0,0.5,1.0]))
coverage_m_union_c2 = hline!([1.00],lw=2,linecolor=:black,linestyle=:dash)
coverage_m_union_c2 = hline!([0.95],lw=2,linecolor=:black,linestyle=:dot)
display(coverage_m_union_c2)
savefig(coverage_m_union_c2,filepath_save[1] * "Fig_" * "coverage_m_union_c2"  * ".pdf")


# Data realisations - r1
coverage_r_r1_c1 = hline([1.00],lw=2,linecolor=:black,linestyle=:dash)
coverage_r_r1_c1 = hline!([0.95],lw=2,linecolor=:black,linestyle=:dot)
coverage_r_r1_c1 = scatter!(t[2:end], sum(r1realisationst_withinbounds_time[:,1,2:end],dims=1)' ./totalseeds, ylim=(0,1.1),markersize=10,msw=0, titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,color=colour1,legend=false,xlab=L"t",ylab=L"C_{z_{i},0.95}^{r_{1}}",xticks=([0,1,2]),yticks=([0,0.5,1.0]))
display(coverage_r_r1_c1)
savefig(coverage_r_r1_c1,filepath_save[1] * "Fig_" * "coverage_r_r1_c1"  * ".pdf")


coverage_r_r1_c2 = hline([1.00],lw=2,linecolor=:black,linestyle=:dash)
coverage_r_r1_c2 = hline!([0.95],lw=2,linecolor=:black,linestyle=:dot)
coverage_r_r1_c2 = scatter!(t[2:end], sum(r1realisationst_withinbounds_time[:,2,2:end],dims=1)' ./totalseeds, ylim=(0,1.1),markersize=10,msw=0, titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,color=colour2,legend=false,xlab=L"t",ylab=L"C_{z_{i},0.95}^{r_{1}}",xticks=([0,1,2]),yticks=([0,0.5,1.0]))
display(coverage_r_r1_c2)
savefig(coverage_r_r1_c2,filepath_save[1] * "Fig_" * "coverage_r_r1_c2"  * ".pdf")

round(mean( [sum(r1realisationst_withinbounds_time[:,1,2:end],dims=1)' ./totalseeds; sum(r1realisationst_withinbounds_time[:,2,2:end],dims=1)' ./totalseeds ] ),digits=3)



# Data realisations - r2
coverage_r_r2_c1 = hline([1.00],lw=2,linecolor=:black,linestyle=:dash)
coverage_r_r2_c1 = hline!([0.95],lw=2,linecolor=:black,linestyle=:dot)
coverage_r_r2_c1 = scatter!(t[2:end], sum(r2realisationst_withinbounds_time[:,1,2:end],dims=1)' ./totalseeds, ylim=(0,1.1),markersize=10,msw=0, titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,color=colour1,legend=false,xlab=L"t",ylab=L"C_{z_{i},0.95}^{r_{2}}",xticks=([0,1,2]),yticks=([0,0.5,1.0]))
display(coverage_r_r2_c1)
savefig(coverage_r_r2_c1,filepath_save[1] * "Fig_" * "coverage_r_r2_c1"  * ".pdf")


coverage_r_r2_c2 = hline([1.00],lw=2,linecolor=:black,linestyle=:dash)
coverage_r_r2_c2 = hline!([0.95],lw=2,linecolor=:black,linestyle=:dot)
coverage_r_r2_c2 = scatter!(t[2:end], sum(r2realisationst_withinbounds_time[:,2,2:end],dims=1)' ./totalseeds, ylim=(0,1.1),markersize=10,msw=0, titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,color=colour2,legend=false,xlab=L"t",ylab=L"C_{z_{i},0.95}^{r_{2}}",xticks=([0,1,2]),yticks=([0,0.5,1.0]))
display(coverage_r_r2_c2)
savefig(coverage_r_r2_c2,filepath_save[1] * "Fig_" * "coverage_r_r2_c2"  * ".pdf")

round(mean( [sum(r2realisationst_withinbounds_time[:,1,2:end],dims=1)' ./totalseeds; sum(r2realisationst_withinbounds_time[:,2,2:end],dims=1)' ./totalseeds ] ),digits=3)


# Data realisations - union
coverage_r_union_c1 = hline([1.00],lw=2,linecolor=:black,linestyle=:dash)
coverage_r_union_c1 = hline!([0.95],lw=2,linecolor=:black,linestyle=:dot)
coverage_r_union_c1 = scatter!(t[2:end], sum(realisationst_withinbounds_time[:,1,2:end],dims=1)' ./totalseeds, ylim=(0,1.1),markersize=10,msw=0, titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,color=colour1,legend=false,xlab=L"t",ylab=L"C_{z_{i},0.95}",xticks=([0,1,2]),yticks=([0,0.5,1.0]))
display(coverage_r_union_c1)
savefig(coverage_r_union_c1,filepath_save[1] * "Fig_" * "coverage_r_union_c1"  * ".pdf")


coverage_r_union_c2 = hline([1.00],lw=2,linecolor=:black,linestyle=:dash)
coverage_r_union_c2 = hline!([0.95],lw=2,linecolor=:black,linestyle=:dot)
coverage_r_union_c2 = scatter!(t[2:end], sum(realisationst_withinbounds_time[:,2,2:end],dims=1)' ./totalseeds, ylim=(0,1.1),markersize=10,msw=0, titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,color=colour2,legend=false,xlab=L"t",ylab=L"C_{z_{i},0.95}",xticks=([0,1,2]),yticks=([0,0.5,1.0]))
display(coverage_r_union_c2)
savefig(coverage_r_union_c2,filepath_save[1] * "Fig_" * "coverage_r_union_c2"  * ".pdf")

round(mean( [sum(realisationst_withinbounds_time[:,1,2:end],dims=1)' ./totalseeds; sum(realisationst_withinbounds_time[:,2,2:end],dims=1)' ./totalseeds ] ),digits=3)

                 

# Data realisations - MLE

coverage_MLE_c1 = hline([1.00],lw=2,linecolor=:black,linestyle=:dash)
coverage_MLE_c1 = hline!([0.95],lw=2,linecolor=:black,linestyle=:dot)
coverage_MLE_c1 = scatter!(t[2:end], sum(realisations_MLE_withinbounds_time[:,1,2:end],dims=1)' ./totalseeds, xlim=(0,2.1), ylim=(0,1.1),markersize=10,msw=0, titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,color=colour1,legend=false,xlab=L"t",ylab=L"C_{z_{i},0.95}",xticks=([0,1,2]),yticks=([0,0.5,1.0]))
display(coverage_MLE_c1)
savefig(coverage_MLE_c1,filepath_save[1] * "Fig_" * "coverage_MLE_c1"  * ".pdf")

round(mean(sum(realisations_MLE_withinbounds_time[:,1,2:end],dims=1)' ./totalseeds),digits=3)


coverage_MLE_c2 = hline([1.00],lw=2,linecolor=:black,linestyle=:dash)
coverage_MLE_c2 = hline!([0.95],lw=2,linecolor=:black,linestyle=:dot)
coverage_MLE_c2 = scatter!(t[2:end], sum(realisations_MLE_withinbounds_time[:,2,2:end],dims=1)' ./totalseeds, xlim=(0,2.1),ylim=(0,1.1),markersize=10,msw=0, titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,color=colour2,legend=false,xlab=L"t",ylab=L"C_{z_{i},0.95}",xticks=([0,1,2]),yticks=([0,0.5,1.0]))
display(coverage_MLE_c2)
savefig(coverage_MLE_c2,filepath_save[1] * "Fig_" * "coverage_MLE_c2"  * ".pdf")

round(mean( sum(realisations_MLE_withinbounds_time[:,2,2:end],dims=1)' ./totalseeds),digits=3)


round(mean( [sum(realisations_MLE_withinbounds_time[:,1,2:end],dims=1)' ./totalseeds; sum(realisations_MLE_withinbounds_time[:,2,2:end],dims=1)' ./totalseeds ] ),digits=3)

