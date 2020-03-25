using DrWatson
quickactivate(@__DIR__,"Random_coding")
#Strutcture and function for 1D model and ML-MSE inference
using Distributions,StatsBase , LinearAlgebra, MultivariateStats,Random,SparseArrays
include(srcdir("network.jl"))

function double_width(N::Int64,σVec::Array{Float64},η::Float64,cVec::Array{Float64},σ2::Float64;nets=4)

    ε = zeros(length(σVec),length(cVec),nets)
    for n=1:nets
        W = randn(N,L)*sqrt(1/L)
        for c= cVec
            ε = zeros(length(σVec))
            for (i,σ1) = enumerate(σVec)
                σp = [σ1,σ2]; np = Network(N,Lp,σp,rxrnorm=0,c=c);
                np.W = W
                ε_series = MSE_net_gon(np, η, MC=1)
                ε[i] = mean(ε_series[end-40,end])
                println(ε[i])
            end
            println(c)
        end
    end
    return ε
end
L,N = 500,25; η = 0.5;
Lp = [250,250]; cVec = collect(0:0.1:1.5);     σVec = collect((1:2:40)/L);
σ2Vec = [1/L, 2/L,5/L]
ε = [double_width(N,σVec,η,cVec,σ2,nets=1) for σ2 = σ2Vec]
σ2min,σ2max = first(σ2Vec),last(σ2Vec);
name = savename("double_width" , (@dict N η σ2min),"jld")
data = Dict("NVec"=>NVec ,"σVec" => σVec,"σ2Vec" => σ2Vec,"cVec" =>cVec,"ε" => ε,)
safesave(datadir("sims/iidnoise/double_width",name) ,data)
