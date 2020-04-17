using DrWatson
quickactivate(@__DIR__,"Random_coding")
#Strutcture and function for 1D model and ML-MSE inference
using Distributions,StatsBase , LinearAlgebra, MultivariateStats,Random,SparseArrays
include(srcdir("network.jl"))

function vary_σ1(W::Array{Float64,2},σVec::Array{Float64},σ2::Float64,η::Float64,c::Float64)
    ε = zeros(length(σVec))
    for (i,σ1) = enumerate(σVec)
        σp = [σ1,σ2]; np = Network(N,Lp,σp,rxrnorm=1,c=c);np.W = W
        U,V = compute_tuning_curves(np,x_test);
        println(mean(var(V,dims=2)))
        ε_series = MSE_net_gon(np, η, MC=1)
        ε[i] = mean(ε_series[end-40,end])
    end
    εo,io = findmin(ε );
    println("With c =",c," and σ2= ",σ2," σ1 opt = ",σVec[io]  ,"ε_o = ",εo)
    return ε
end


function double_width(N::Int64,σVec::Array{Float64},η::Float64,cVec::Array{Float64},σ2::Float64;nets=4)
    ε = zeros(length(σVec),length(cVec),nets)
    Threads.@threads for net=1:nets
        #W = randn(N,L)*sqrt(1/L);
        W = randn(N,Lp[1])*sqrt(1/L); W = hcat(W,W)
        for (i,c)=enumerate(cVec)
            @time ε[:,i,net] = vary_σ1(W,σVec,σ2,η,c)
        end
    end
    return ε
end
L,N = 500,25; η = 0.5;Lp = [250,250];
cVec =[0.,0.1,0.5,1.,1.5,2.,5.,10.];     σVec = collect((1:2:40)/L);
σ2Vec = [3/L,5/L,7/L,9/L]
ε = [double_width(N,σVec,η,cVec,σ2,nets=1) for σ2 = σ2Vec]
σ2min,σ2max = first(σ2Vec),last(σ2Vec);
name = savename("double_width" , (@dict N η σ2min σ2max),"jld")
data = Dict("σVec" => σVec,"σ2Vec" => σ2Vec,"cVec" =>cVec,"ε" => ε,)
safesave(datadir("sims/iidnoise/double_width",name) ,data)
