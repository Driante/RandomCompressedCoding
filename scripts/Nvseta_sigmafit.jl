using DrWatson
quickactivate(@__DIR__,"Random_coding")
using JLD
include(srcdir("network3D.jl"))
include(srcdir("lalazarabbott.jl"))
#Coding properties of network at best σ
function singleσ(V,N::Int64,V_l::Array{Float64},η::Float64;nets=4)
    #Compute error for different number of neurons and compare with case where tuning curves are linear
    ε = zeros(nets) ; ε_l = zeros(nets)
    Nneurons,ntest = size(V)
     @time Threads.@threads for net=1:nets
         println("On thread n ", Threads.threadid())
        Nid =  rand(1:Nneurons,N)
        ε_serie = MSE_net_gon(V[Nid,:],η,x_test); ε[net] = sqrt(mean(ε_serie[end-30,end]))
        ε_l_serie  = MSE_net_gon(V_l[Nid,:],η,x_test); ε_l[net] = sqrt(mean(ε_l_serie[end-30,end]))
        println("N=",N," η= ",η )
    end
    println(" ε_o=",  mean(ε)," ε_l=" , mean(ε_l))
    return ε,ε_l
end
function singleσ(V,N::Int64,V_l::Array{Float64},ηVec::Array{Float64};nets=8)
    #Compute error for different number of neurons and compare with case where tuning curves are linear
    ε = zeros(nets) ; ε_l = zeros(nets)
    Nneurons,ntest = size(V)
     @time Threads.@threads for net=1:nets
         println("On thread n ", Threads.threadid())
        Nid =  rand(1:Nneurons,N);
        ε_serie = MSE_net_gonD(V[Nid,:],ηVec[Nid],x_test);    ε[net] = sqrt(mean(ε_serie[end-30,end]))
        ε_l_serie  = MSE_net_gonD(V_l[Nid,:],ηVec[Nid],x_test); ε_l[net] = sqrt(mean(ε_l_serie[end-30,end]))
        println("N=",N)
    end
    println(" ε_o=",  mean(ε)," ε_l=" , mean(ε_l))
    return ε,ε_l
end

σf = 22.
#Use noise from data
r,r_var ,tP,η,trials,r_stab,η_stab= import_and_clean_data(stab_var=1);
N,~ = size(r)
ηVec =vec(mean(var(η_stab,dims=3)./nanvar(r_stab,2),dims=2))
#Import precomputed tuning curves
data = load(datadir("sims/LalaAbbott/tuning_curves","tuning_curves3D_ntest=9261_s=0.1.jld"))
Vdict,σVec,x_test= data["Vdict"],data["σVec"],data["x_test"];
V = Vdict[σf]./std(Vdict[σf],dims=2); V = V[1:N,:];
#Build Linear tuning curves
V_l,PP,R2 = linear_fit(Vdict[last(σVec)],x_test); V_l = vcat(V_l'...);
V_l ./= sqrt.((var(V_l,dims=2))); V_l  .-=mean(V_l,dims=2); V_l = V_l[1:N,:]
NVec = Int.(round.(10 .^(1.3:0.1:2.3)));

#Fix noise variance
ηVec  = 1.0:1.0:5.
ε =[[singleσ(V,N,V_l,PP,η)    for N =NVec] for η=ηVec];


 #ε = [singleσ(V,N,V_l,ηVec) for N=NVec]

Nmin,Nmax = first(NVec),last(NVec);η_min, η_max = minimum(ηVec),maximum(ηVec)

name = savename("Nvsetasingle" , (@dict Nmin Nmax η_min η_max σf),"jld")
data = Dict("NVec"=>NVec ,"σVec" => σVec,"ηVec" => ηVec,"ε" => ε)
safesave(datadir("sims/LalaAbbott/iidnoise",name) ,data)
