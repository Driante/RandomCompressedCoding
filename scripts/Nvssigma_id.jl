using DrWatson
quickactivate(@__DIR__,"Random_coding")
include(srcdir("network.jl"))
#Computation of curves for error in function of σ for different network realization and different values of N.
#Gaussian noise distribution and ideal decoder
function vary_σ(n::Network,σVec::Array{Float64},η::Float64,fdec::Function;ntrial=50,MC=0)
    N,L,W = n.N,n.L,n.W;ε = zeros(length(σVec));
    for (i,σi) = enumerate(σVec)
        #Keep fixed synaptic matrix
        n2 = Network(N,L,σi,f1=n.f1,rxrnorm=0); n2.W = W
        if MC==0
            ε[i] = sqrt(fdec(n2,η,ntrial=ntrial))
        else
            ε_serie = fdec(n2,η,MC=MC)
            ε[i] = sqrt(mean(ε_serie[end-50,end]))
            println(σi)
        end
    end
    return ε
end
function Nvsσ(N::Int64,σVec::Array{Float64},η::Float64; nets = 4,circ=0,MC=0)
    σs = 10/L;ε = zeros(length(σVec),nets);
    if circ ==0
        f1 = gaussian; #fdec= MSE_ideal_gon
        fdec = MSE_net_gon
    else
        f1= VonMises; fdec = MSE_ideal_gon_c
    end
    Threads.@threads for net=1:nets
        n= Network(N,L,σs,f1=f1); ε[:,net]= vary_σ(n,σVec,η,fdec,MC=MC);
    end
    println("N=",N)
    return ε
end

L=500
NVec = 10:5:60;σVec = collect(1.:2:50.)/L;
#η = 0.5:0.5:0.6;
ηVec = 0.1:0.1:1.5; NVec = Int.(round.(10 .^(1:0.1:2.5)))
circ=0
ε= [[Nvsσ(N,σVec,η,circ=0,MC=1) for N=NVec] for η = ηVec]
#Do the same for a set of different noise levels
#ηVec = 0.1:0.1:1.5; NVec = Int.(round.(10 .^(1:0.1:2.5)))
Nmin,Nmax = first(NVec),last(NVec);η_min, η_max = first(ηVec),last(ηVec)
name = savename("Nvssigma" , (@dict Nmin Nmax η_min η_max circ),"jld")
data = Dict("NVec"=>NVec ,"σVec" => σVec,"ηVec" => ηVec,"ε" => ε,)
safesave(datadir("sims/iidnoise/idealdec",name) ,data)
