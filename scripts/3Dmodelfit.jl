using DrWatson
quickactivate(@__DIR__,"Random_coding")
using JLD, MultivariateStats
include(srcdir("lalazarabbott.jl"))
#Script that save the quantity relevant for fititng procedure of the tuning curves from data
#Data statistics and summary of fitting procedure
r,r_var ,tP= import_and_clean_data()
#Fit response with a linear model, obtaining principal vectors and R2; Compute histogram of R2
r_linear, PP, R2 = linear_fit(r , tP); r_linear = (hcat(r_linear...)' )[:,:] ;
#Fill NaN response with corresponding linear model + Standardize firing rate, to fit with our model
r[isnan.(r)] = r_linear[isnan.(r)]; r_linear .-= mean(r_linear,dims=2); r_linear ./=sqrt.(var(r_linear,dims=2))
#Standardize also the true data
r .-=mean(r,dims=2); r./=std(r,dims=2);
#The complexity is computed on the standradized tuning curves, to compare different range of firing rates
complexity_data = complexity_opt(r,tP,36);complexity_linear = complexity_opt(r_linear,tP,36);
data_pca = fit(PCA,r,pratio=0.999)
name = savename("3D_fitting_data" , (@dict   ),"jld")
data = Dict("complexity_data"=>complexity_data,"complexity_linear"=>complexity_linear,"R2" => R2,"data_pca" => data_pca)
safesave(datadir("sims/LalaAbbott/tuning_curves",name) ,data)
