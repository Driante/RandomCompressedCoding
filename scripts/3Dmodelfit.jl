using DrWatson
quickactivate(@__DIR__,"Random_coding")
using JLD
using Plots,Flux
pyplot()
include(srcdir("lalazarabbott.jl"))
include(srcdir("RM_model3D.jl"))

r,r_var ,tP= import_and_clean_data()
r = (r.-nanmean(r,2))./sqrt.(nanvar(r,2))
#Fit response with a linear model, obtaining principal vectors and R2; Compute histogram of R2
r_linear, PP, R2 = linear_fit(r , tP); r_linear = (hcat(r_linear...)' )[:,:] ;  hR2 = fit(Histogram,R2,0:0.1:1);
#Fill NaN response with corresponding linear model + Standardize firing rate, to fit with our model
r[isnan.(r)] = r_linear[isnan.(r)]; r_linear = standardize(ZScoreTransform,r_linear)

np= 17;s = 0.1
Δi = range(-40.,40., length=np); tP_fine = vcat(reshape([[x,y,z] for x=Δi,y=Δi,z=Δi],:,1)'...)
σVec= 3.:1.:35.
r_sf = zeros(N,np^3,length(σVec))
N=460; L=100; M=L^3; x_min  = -100.; x_max = 100.;
a=1;σmax=36.;σmin=15. ; θ = (-σmax^(-a)+σmin^(-a))^(1/a)
σd = Truncated(Pareto(a,θ),σmin,σmax);
σ = 23.0
myn = Network3D(M,N,σ, x_min,x_max,sparsity=0.1)
r_s = myn(tP[1]);r_s ./= std(r_s,dims=2)
#Statistic of generated network
r_linear_s,PP_s, R2_s = linear_fit(r_s,repeat([tP[1]],N));
r_s[isnan.(r_s)].=0
