using DrWatson
quickactivate(@__DIR__,"Random_coding")
using JLD
include(srcdir("network3D.jl"))
#Script comparing the mean square error in the pure and conjuntive case for a 3D stimulus


function vary_σ(Vdict::Dict,Nid::Array{Int64},η::Float64,σVec)
    nσ = length(σVec);ε = zeros(nσ)
    for (i,σ) = enumerate(σVec)
        V = Vdict[σ][Nid,:];
        @time ε_serie = MSE_net_gon(V,η,x_test)
        ε[i] = sqrt(mean(ε_serie[end-30,end]))
    end
    return ε
end

function Nvsσ(Vpdict::Dict,Vcdict::Dict,N::Int64,σVec,η::Float64;nets=4)
    #Compute error for different number of neurons and compare with case where tuning curves are linear
    εp,εc = [zeros(length(σVec),nets) for n=1:2]
    Nneurons,ntest = size(first(Vcdict)[2])
     @time Threads.@threads for net=1:nets
        println("On thread n ", Threads.threadid())
        Nid =  rand(1:Nneurons,N)
        εp[:,net] = vary_σ(Vpdict,Nid,η,σVec)
        εc[:,net] = vary_σ(Vcdict,Nid,η,σVec)
        println("N=",N," η= ",η)
    end
    println(" εp=",  minimum(εp)," εc=" , minimum(εc))
    return εp,εc
end

data =load(datadir("sims/LalaAbbott/tuning_curves","tuning_curves3D_pvsc_Meq_ntest=20.jld"))
Vcdict,Vpdict,x_test,σVec = data["Vcdict"],data["Vpdict"],data["x_test"],data["σVec"]
η =1.0; NVec = Int.(round.(10 .^(1:0.1:2.1)))
ε = [Nvsσ(Vpdict,Vcdict,N,σVec,η) for N=NVec]
Nmin,Nmax = first(NVec),last(NVec);
name = savename("Nvssigma3D_cvsp" , (@dict Nmin Nmax η),"jld")
data = Dict("NVec"=>NVec ,"σVec" => σVec,"ε" => ε)
safesave(datadir("sims/LalaAbbott/iidnoise",name) ,data)
