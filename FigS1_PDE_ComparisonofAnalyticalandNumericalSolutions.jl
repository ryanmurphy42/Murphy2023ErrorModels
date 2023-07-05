# Murphy et al. (2023) 
# Implementing measurement error models in a likelihood-based framework for 
# estimation, identifiability analysis, and prediction in the life sciences

# FigS1_PDE_ComparisonAnalyticalandNumericalSolution

#######################################################################################
## Initialisation including packages, plot options, filepaths to save outputs

using Plots
using LaTeXStrings
using DifferentialEquations
using ModelingToolkit # to solve pde
using MethodOfLines # to solve pde
using DomainSets # to solve pde
using SpecialFunctions # for erf

pyplot() # plot options
fnt = Plots.font("sans-serif", 20) # plot options
global cur_colors = palette(:default) # plot options
isdir(pwd() * "\\FigS1_PDE_ComparisonAnalyticalandNumericalSolution\\") || mkdir(pwd() * "\\FigS1_PDE_ComparisonAnalyticalandNumericalSolution") # make folder to save figures if doesnt already exist
filepath_save = [pwd() * "\\FigS1_PDE_ComparisonAnalyticalandNumericalSolution\\"] # location to save figures "\\Fig2\\"] # location to save figures



#######################################################################################
## Mathematical model

a = zeros(3); # three parameters, D, r1, r2
D=0.5;
r1=1.2;
r2=0.8;
t_record = [0.001, 0.25, 0.5, 0.75, 1.0]
t_max = maximum(t_record)
L = 10; # domain length

#######################################################################################
## Solving the mathematical model numerically

@parameters x t
    @variables u(..) v(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

function modelnumerical(t_record,a)
    # PDE 
    D=a[1];
    r1=a[2];
    r2=a[3];

    eq = [Dt(u(x,t)) ~ D*Dxx(u(x,t)) - r1*u(x,t),
    Dt(v(x,t)) ~ D*Dxx(v(x,t)) + r1*u(x,t) - r2*v(x,t)]

    bcs = [u(x,0) ~ 100 * (x < 1.)*(x > -1. ), # 100 for -1 < x < 1
    Dx(u(-L,t)) ~ 0,
    Dx(u(L,t)) ~ 0,
    v(x,0) ~ 0,
    Dx(v(-L,t)) ~ 0,
    Dx(v(L,t)) ~ 0]

    # bcs = [u(x,0) ~ 100 * (x < 1.)*(x > -1. ), # 100 for -1 < x < 1
    # v(x,0) ~ 0]

    domains = [ t ∈ Interval(0,t_max),
        x ∈ Interval(-L,L)]
    @named pdesys = PDESystem(eq,bcs,domains,[x,t],[u(x,t),v(x,t)])

    N=151;
    order = 2 # This may be increased to improve accuracy of some schemes
    discretization = MOLFiniteDifference([x=>N], t, approx_order=order)

    println("Discretization:")
    @time prob = discretize(pdesys,discretization) # Converts the PDE problem into an ODE problem 
    println("Solve:")
    @time modelsol = solve(prob, TRBDF2(), saveat=t_record)

    return modelsol
end

solnumerical = modelnumerical(t_record,[D,r1,r2])

discrete_x = solnumerical[x]
discrete_t = solnumerical[t]

solnumericalu = solnumerical[u(x, t)]
solnumericalv = solnumerical[v(x, t)]




#######################################################################################
## Solving the mathematical model analytically

a = [D,r1,r2]

function modelanalytical(t_record,a)
    # PDE 
    D=a[1];
    r1=a[2];
    r2=a[3];
    h=1.0;
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


solanalytical = modelanalytical(t_record,[D,r1,r2])

solanalyticalu = solanalytical[1,:,:]
solanalyticalv = solanalytical[2,:,:]





#######################################################################################
## Plots comparing the analytical and numerical solutions

colour1=:green2
colour2=:magenta

for k in 1:length(discrete_t)
    p0 = plot(discrete_x,solnumericalu[1:end,k],lw=3,linecolor=colour1) # 2:end since end = 1, periodic condition
    p0 = plot!(p0,discrete_x,solnumericalv[1:end,k],xlab=L"x",ylab=L"c_{1}(x,t), c_{2}(x,t)",lw=3,legend=false,titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt,linecolor=colour2) # 2:end since end = 1, periodic condition

    p0 = plot!(discrete_x,solanalyticalu[1:end,k],lw=3,linecolor=:black,linestyle=:dash) # 2:end since end = 1, periodic condition
    p0 = plot!(p0,discrete_x,solanalyticalv[1:end,k],lw=3,linecolor=:black,linestyle=:dash,ylims=(0,105)) # 2:end since end = 1, periodic condition

    display(p0)
    # savefig(p0,filepath_save[1] * "Fig0_loop" * string(k) * ".pdf")
end