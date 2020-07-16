using DrWatson
quickactivate(@__DIR__,"Random_coding")
using JLD
include(srcdir("network.jl"))
function compare_covariance(n::Network,σVec::Array{Float64},η::Float64,ηu::Float64;ntrial=50,MC=1)
    N,L,W = n.N,n.L,n.W;
    εc,εi,εr= [zeros(length(σVec)) for n=1:3];
    X = sqrt(1/L)*randn(N,L);
    for (i,σi) = enumerate(σVec)
        #Keep fixed synaptic matrix
        n2 = Network(N,L,σi,rxrnorm=0); n2.W = W;
        ηu_scaled = ηu/n2.A^2
        #Covariance matrix for input noise
        Σ = ηu_scaled*n2.A^2*n2.W*n2.W' + η*I;
        εc_serie= MSE_net_ginoutn(n2,Symmetric(Σ),tol=1E-8);
        #Covariance matrix without off diagonal terms
        Σd = Diagonal((ηu_scaled*n2.A^2+η)*ones(N))
        εi_serie = MSE_net_ginoutn(n2,Σd,tol=1E-8);
        #Random covariance matrix
        εr_serie = MSE_net_ginoutn(n2,Symmetric(η*I + ηu_scaled*n2.A^2*X*X'),tol=1E-8)
        εc[i],εi[i], εr[i] =sqrt(mean(εc_serie[end-50,end])),sqrt(mean(εi_serie[end-50,end])),sqrt(mean(εr_serie[end-50,end]))
    end
    return εc,εi,εr
end
function Nvsσ_cvsi(N::Int64,σVec::Array{Float64},η::Float64,ηu::Float64; nets = 4,circ=0,MC=1)
    σs = 10/L;
    εc, εi , εr = [zeros(length(σVec),nets) for n=1:3]
    Threads.@threads for net=1:nets
        n= Network(N,L,σs);
        εc[:,net],εi[:,net], εr[:,net]  = compare_covariance(n,σVec,η,ηu);
        @info "Finished N= $(N),  η= $(ηu)   on thread $(Threads.threadid()): εc/εi = $(εc[:,net]./εi[:,net])"
    end
    return εc, εi, εr
end

L=500
NVec = [70];σVec = collect(1.:3:50.)/L;
η = 0.01; ηuVec = [.5]
ε= [[@time Nvsσ_cvsi(N,σVec,η,ηu) for N=NVec] for ηu = ηuVec]
Nmin,Nmax = first(NVec),last(NVec);ηu_min, ηu_max = first(ηuVec),last(ηuVec)
name = savename("Nvssigma" , (@dict Nmin Nmax η ηu_min ηu_max ),"jld")
data = Dict("NVec"=>NVec ,"σVec" => σVec,"ηuVec" => ηuVec,"ε" => ε)
safesave(datadir("sims/iidnoise/inputnoise",name) ,data)
