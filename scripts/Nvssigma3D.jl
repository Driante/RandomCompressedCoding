using DrWatson
quickactivate(@__DIR__,"Random_coding")
using JLD
include(srcdir("network3D.jl"))
include(srcdir("lalazarabbott.jl"))
function vary_σ(Vdict::Dict,Nid,η::Float64,σVec::Array{Float64})
    nσ = length(σVec);ε = zeros(nσ)
    for (i,σ) = enumerate(σVec)
        V = Vdict[σ][Nid,:];
        @time ε_serie = MSE_net_gon(V,η,x_test)
        ε[i] = sqrt(mean(ε_serie[end-30,end]))
    end
    return ε
end

function Nvsσ(Vdict::Dict,N::Int64,σVec::Array{Float64},V_l::Array{Float64},PP::Array{Float64},η::Float64;nets=4)
    #Compute error for different number of neurons and compare with case where tuning curves are linear
    ε = zeros(length(σVec),nets) ; ε_l = zeros(nets)
     @time Threads.@threads for net=1:nets
         println("On thread n ", Threads.threadid())
        Nid =  rand(1:412,N)
        ε[:,net] = vary_σ(Vdict,Nid,η,σVec)
        ε_l_serie  = MSE_net_linear(V_l[Nid,:],η,x_test,PP[Nid,:]);
        ε_l[net] = sqrt(mean(ε_l_serie[end-30,end]))
        println("N=",N," η= ",η)
    end
    println(" ε_o=",  mean(ε)," ε_l=" , mean(ε_l))
    return ε,ε_l
end
function singleσ(V,N::Int64,V_l::Array{Float64},PP::Array{Float64},η::Float64;nets=4)
    #Compute error for different number of neurons and compare with case where tuning curves are linear
    ε = zeros(nets) ; ε_l = zeros(nets)
     @time Threads.@threads for net=1:nets
         println("On thread n ", Threads.threadid())
        Nid =  rand(1:412,N)
        ε_serie = MSE_net_gon(V[Nid,:],η,x_test); ε[net] = sqrt(mean(ε_serie[end-30,end]))
        ε_l_serie  = MSE_net_linear(V_l[Nid,:],η,x_test,PP[Nid,:]); ε_l[net] = sqrt(mean(ε_l_serie[end-30,end]))
        println("N=",N," η= ",η )
    end
    println(" ε_o=",  mean(ε)," ε_l=" , mean(ε_l))
    return ε,ε_l
end

#import data of precomputed tuning curves
data = load(datadir("sims/LalaAbbott/tuning_curves","tuning_curves3D_ntest=9261_s=0.1.jld"))
Vdict,σVec,x_test= data["Vdict"],data["σVec"],data["x_test"];
#Normalize row by row the tuning curves
for σ=σVec
    Vdict[σ] ./= std(Vdict[σ],dims=2)
end
#Generate Linear Tuning curves from fitting, standardize also them
V_l,PP,R2 = linear_fit(Vdict[last(σVec)],x_test); V_l = vcat(V_l'...);
V_l ./= sqrt.((var(V_l,dims=2))); V_l  .-=mean(V_l,dims=2)
NVec = 50:20:200; ηVec  = 1.0:1.0:5.0;
#curve for all the σ
#ε =[[Nvsσ(Vdict,N,collect(σVec[1:2:end]),V_l,PP,η) for N =NVec] for η=ηVec];

ε =[[singleσ(Vdict[19.],N,V_l,PP,η)    for N =NVec] for η=ηVec];

Nmin,Nmax = first(NVec),last(NVec);η_min, η_max = first(ηVec),last(ηVec)
name = savename("Nvssigma3D" , (@dict Nmin Nmax η_min η_max),"jld")
data = Dict("NVec"=>NVec ,"σVec" => σVec,"ηVec" => ηVec,"ε" => ε)
safesave(datadir("sims/LalaAbbott/iidnoise",name) ,data)
