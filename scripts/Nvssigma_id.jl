using DrWatson
quickactivate(@__DIR__,"Random_coding")
using Distributions,StatsBase , LinearAlgebra, MultivariateStats,Random,SparseArrays
include(srcdir("network.jl"))
#Computation of curves for error in function of σ for different network realization and different values of N
function optimal_sigma(n::Network,σVec::Array{Float64},η::Float64,fdec::Function;ntrial=50,MC=0)
    ε = zeros(length(σVec)); x_t = repeat(x_test,ntrial)
    W = n.W
    for (i,σi) = enumerate(σVec)
        #Keep fixed synaptic matrix
        n2 = Network(N,L,σi,f1=n.f1,rxrnorm=0); n2.W = W
        if MC=0
            ε[i] = sqrt(fdec(n2,η,ntrial=ntrial))
        else
            ε_serie = fdec(n2,η,MC=MC)
            ε[i] = sqrt(mean(ε_series[end-50,end]))
    end
    ε_o,i_o= findmin(ε); σ_o = σVec[i_o]
    return ε,ε_o,σ_o
end
function Nvsσ(NVec::Array{Float64},σVec::Array{Float64},η::Float64;L=500, nets = 4,circ=0)
    ε = zeros(length(NVec),length(σVec),nets); ε_o = zeros(length(NVec),nets);
    σ_o = zeros(length(NVec),nets); σs = 10/L
    if circ ==0
        f1 = gaussian; fdec= MSE_ideal_gon
    else
        f2= VonMises; fdec = MSE_ideal_gon_c
    end
    for N = NVec
        for net=1:nNet
            n= Network(N,L,σs,f1=f1);
            ε = optimal_sigma(n,σVec,η,fdec)
        end
    end
end
