using DrWatson
quickactivate(@__DIR__,"Random_coding")
using JLD
include(srcdir("network.jl"))
function vary_σ(n::Network,σVec::Array{Float64},η::Float64,ηu::Float64;ntrial=50,MC=1)
    N,L,W = n.N,n.L,n.W;εc,εi= [zeros(length(σVec)) for n=1:2];
    for (i,σi) = enumerate(σVec)
        #Keep fixed synaptic matrix
        n2 = Network(N,L,σi,rxrnorm=0); n2.W = W;
        ηu_scaled = ηu/n2.A^2
        Σ = ηu_scaled*n2.A^2*n2.W*n2.W' + η*I; iΣ = inv(Σ); ηeq = mean(diag(Σ))
        @info "ηeq = $(ηeq)"
        εc_serie= MSE_net_ginoutn(n2,η,ηu_scaled);
        εi_serie = MSE_net_gon(n2,ηeq,MC=MC)
        εc[i],εi[i] =sqrt(mean(εc_serie[end-50,end])),sqrt(mean(εi_serie[end-50,end]))
    end
    return εc,εi
end
function Nvsσ_cvsi(N::Int64,σVec::Array{Float64},η::Float64,ηu::Float64; nets = 4,circ=0,MC=1)
    σs = 10/L;εc, εi = [zeros(length(σVec),nets) for n=1:2]
    Threads.@threads for net=1:nets
        n= Network(N,L,σs); εc[:,net],εi[:,net]  = vary_σ(n,σVec,η,ηu,MC=MC);
        @info "Finished N= $(N),  η= $(ηu)   on thread $(nThreads.threadid())"
    end
    return εc,εi
end

L=500
NVec = 10:5:60;σVec = collect(1.:2:50.)/L;
η = 0.01; ηuVec = 0.1:0.3:1
ε= [[@time Nvsσ_cvsi(N,σVec,η,ηu) for N=NVec] for ηu = ηuVec]
Nmin,Nmax = first(NVec),last(NVec);ηu_min, ηu_max = first(ηuVec),last(ηuVec)
name = savename("Nvssigma" , (@dict Nmin Nmax η ηu_min ηu_max ),"jld")
data = Dict("NVec"=>NVec ,"σVec" => σVec,"ηVec" => ηVec,"ε" => ε,)
safesave(datadir("sims/iidnoise/inputnoise",name) ,data)
