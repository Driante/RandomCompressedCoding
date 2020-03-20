#Analysis and experiments with Lalazar Abbott data
using DrWatson
quickactivate(@__DIR__,"Random_coding")
using JLD
#include(srcdir("lalazarabbott.jl"))
include(srcdir("RM_model3D.jl"))
N=412; L=100; M=L^3; x_min  = -100.; x_max = 100.;σ = 24.
#Use data to fix the variance of the firing rates
myn = Network3D(M,N,σ, x_min,x_max,sparsity=0.1)
# r_s = myn(tP[1]);
#Statistic of generated network
# r_linear_s,PP_s, R2_s = linear_fit(r_s,repeat([tP[1]],N)); hR2_s = fit(Histogram,R2_s,0:0.1:1);
# complexity_s = vec(complexity(r_s,repeat([tP[1]],N),36));
# hcmpx_s = fit(Histogram,complexity_s,nbins=20);
#Generate Extended data at intermediate positions
np= 10;s = 0.1
Δi = range(-40.,40., length=np); tP_fine = vcat(reshape([[x,y,z] for x=Δi,y=Δi,z=Δi],:,1)'...)
σVec= 3.:1.:35.
r_sf = zeros(N,np^3,length(σVec))
for (i,σ) = enumerate(σVec)
    myn = Network3D(M,N,σ, x_min,x_max,sparsity=s)
    r_sf[:,:,i] = myn(tP_fine);myn.Z *= mean(vec(std(r_sf[:,:,i],dims=2)))
    r_sf[:,:,i] ./= mean(vec(std(r_sf,dims=2)))
    # r_linear_sf,PP_sf,R2_sf = linear_fit(r_s,repeat([tP_fine],N))
    # hR2_sf = fit(Histogram,R2_sf,0:0.1:1);
    # complexity_sf = vec(complexity(r_s,repeat([tP_fine],N),9));
    # hcmpx_sf = fit(Histogram,complexity_sf,nbins=20);
    println(σ)
end
name = savename("r_sf" , (@dict np  s ),"jld")
data = Dict("r_sf"=>r_sf ,"σVec" => σVec,"tP_fine" => tP_fine)
safesave(datadir("sims/LalaAbbott",name) ,data)
