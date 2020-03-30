using DrWatson
quickactivate(@__DIR__,"Random_coding")
using JLD
pyplot()
include(srcdir("lalazarabbott.jl"))
#Script that output the complexity and R2 histogram of the data, their linear fit, and the simulations
#Data statistics and summary of fitting procedure
r,r_var ,tP= import_and_clean_data()
#Fit response with a linear model, obtaining principal vectors and R2; Compute histogram of R2
r_linear, PP, R2 = linear_fit(r , tP); r_linear = (hcat(r_linear...)' )[:,:] ;
#Fill NaN response with corresponding linear model + Standardize firing rate, to fit with our model
r[isnan.(r)] = r_linear[isnan.(r)]; r_linear .-= mean(r_linear,dims=2); r_linear ./=sqrt.(var(r_linear,dims=2))
#Standardize also the true data
r .-=mean(r,dims=2); r./=std(r,dims=2);
#The complexity is computed on the standradized tuning curves, to compare different range of firing rates
complexity_data = complexity_opt(r,tP,36);complexity_linear = complexity_opt(r,tP,36)
data = load(datadir("sims/LalaAbbott/tuning_curves","tuning_curves3D_ntest=27_s=0.1.jld"))
Vdict,σVec,x_test = data["Vdict"],data["σVec"],data["x_test"];
N = 412;
#COmpute the same quantities for all the precomputed tuning curves (to the same stimulus)
complexity_sims,R2_sims = [zeros(N,length(σVec)) for n=1:2]
for (i,σ) = enumerate(σVec)
    V = Vdict[σ][2];
    complexity_sims[:,i] = complexity_opt(V,x_test,36); ~,~,R2_sims[:,i] =  linear_fit(V,x_test)
end

name = savename("3D_fitting" , (@dict  ),"jld")
data = Dict("complexity_data"=>complexity_data,"complexity_linear"=>complexity_linear,"complexity-sims" => complexity_sims,"R2" => R2,"R2_sims"=> R2_sims)
safesave(datadir("sims/LalaAbbott/tuning_curves",name) ,data)
