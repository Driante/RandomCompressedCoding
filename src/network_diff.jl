using DrWatson
quickactivate(@__DIR__,"Random_coding")
#Strutcture and function for 1D model and ML-MSE inference, fully differentiable such that is possible to take the gradient with Automatic differentation
using Distributions,StatsBase , LinearAlgebra, MultivariateStats,Random,SparseArrays
using Zygote
include("network.jl")
#Rewrite some of the network function to be differentiable using AD
function compute_tuning_curves_diff(n::Network)
    U = n.f1(x_test',n.A,n.σ,n.cVec)
    V = n.f2.(n.W*U)./n.Z
    n.Z = sqrt.(var(V,dims=2))
    V = V./n.Z
    return U,V
end
function MSE_net_gon(n::Network,η::Float64;ntrial=30)
    N = n.N;
    x_t = vcat([x_test for t=1:ntrial]...)
    U,V = compute_tuning_curves_diff(n)
    R = hcat([V + sqrt(η)*randn(N,length(x_test)) for t=1:ntrial]...)
    H = exp.((V'*R)/(η));Zh = sum(H,dims=1);
    H = H./Zh
    x_ext = H'*x_test
    return mean((x_ext-x_t).^2)
end
function loss(σi,η)
    n.σ=σi;
    return MSE_net_gon(n,η)
end
function optimal_sigma(n::Network,σVec::Array{Float64},η::Float64,fdec::Function;ntrial=50,MC=0)
    ε = zeros(length(σVec)); x_t = repeat(x_test,ntrial)
    for (i,σi) = enumerate(σVec)
        n.σ = σi;
        ε[i] = sqrt(fdec(n,η,ntrial=ntrial))
    end
    ε_o,i_o= findmin(ε); σ_o = σVec[i_o]
    return ε,ε_o,σ_o
end

function find_opt_σ(n::Network,η;ntrial=20)
    σi = n.σ;σ_ev = [σi];ε = [loss(σi,η)]
    for t=1:100
        g = gradient(σi -> loss(σi,η),σi)
        σi -= 0.01*g[1]
        push!(σ_ev,σi)
        println(L*σ_ev[t])
        if mod(t,10) ==0
            push!(ε,MSE_net_gon(n,η,ntrial=100))
        end
    end
    return σ_ev,ε
end

Zygote.refresh()
L=500; N=25;η=0.5;
σi=10/L; n = Network(N,L,σi)
#x_t = repeat(x_test,100);
σVec =  collect((1.:2:40.)/L)
ε,ε_o,σ_o = optimal_sigma(n,σVec,η,MSE_net_gon,ntrial=100)
n.σ = 10/L
σ_ev,ε = find_opt_σ(n,η)
