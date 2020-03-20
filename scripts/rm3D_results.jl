#Analysis and experiments with Lalazar Abbott data
using DrWatson
quickactivate(@__DIR__,"Random_coding")
using JLD
include(srcdir("lalazarabbott.jl"))
include(srcdir("RM_model3D.jl"))
function Nvsσ3D(r,NVec,η,tP;nNet=4,ntrial=20)
    Nneurons,L,nσ = size(r)
    RMSE = zeros(length(NVec),nσ,nNet)
    RMSE_l = zeros(length(NVec),nNet)
    r_l,PP,R2 = linear_fit(r[:,:,end],repeat([tP],Nneurons));r_l = vcat(r_l'...)
    r_l .-= mean(r_l,dims=2); r_l ./= std(r_l,dims=2)
    for (i,N) = enumerate(NVec)
         Threads.@threads for net=1:nNet
            neurons = rand(1:Nneurons,N)
            RMSE_l[i,net] = evaluate_performance_linear(r_l[neurons,:],η,tP,PP[neurons,:],ntrial)
            println(N, " ",RMSE_l[i,net])
            # for j=1:nσ
            #     @time RMSE[i,j,net] = evaluate_performance_netMMSE(r[neurons,:,j],η,tP,ntrial)
            #     println(N," ",net," ",j)
            # end
        end
    end
    return RMSE,RMSE_l
end
function Nvsη3D(r,NVec,ηVec,tP,σj;nNet=4,ntrial=20)
    Nneurons,L,nσ = size(r)
    RMSE = zeros(length(NVec),length(ηVec),nNet)
    RMSE_l = zeros(length(NVec),length(ηVec),nNet)
    r_l,PP,R2 = linear_fit(r[:,:,end],repeat([tP],Nneurons));r_l = vcat(r_l'...)
    r_l .-= mean(r_l,dims=2);
    #r_l ./= std(r_l,dims=2)
    for (i,N) = enumerate(NVec)
          for net=1:nNet
            neurons = rand(1:Nneurons,N)
            for (j,η)=enumerate(ηVec)
                RMSE[i,j,net] = evaluate_performance_ideal3D(r[neurons,:,σj],η,tP,ntrial)
                RMSE_l[i,j,net] = evaluate_performance_linear(r_l[neurons,:],η,tP,PP[neurons,:],ntrial)
                println(N," ",net," ",j)
            end
        end
    end
    return RMSE,RMSE_l
end

data = load(datadir("sims/LalaAbbott","r_sf_np=21_s=0.1.jld"))
r,σVec,tP = data["r_sf"],data["σVec"],data["tP_fine"];
#r .-= mean(r,dims=2)
r ./= std(r,dims=2)
NVec = 10:20:210; η=2.;
# #Brush σ
# RMSE,RMSE_l= Nvsσ3D(r[:,:,1:2:end],NVec,η,tP,nNet=4,ntrial=20)
# Nmin,Nmax = first(NVec),last(NVec); dec="net"
# name = savename("Nsigma3D_linear" , (@dict Nmin Nmax η dec ),"jld")
# data_perf = Dict("NVec"=>NVec ,"σVec" => σVec,"RMSE" => RMSE,"RMSE_l" => RMSE_l)
# safesave(datadir("sims/LalaAbbott",name) ,data_perf)

#Fix σ
σj  =10;
ηVec = 0.5:0.5:7.
RMSE,RMSE_l =Nvsη3D(r,NVec,ηVec,tP,σj,nNet=4,ntrial=10)
Nmin,Nmax = first(NVec),last(NVec)
name = savename("Nvseta3D_linear" , (@dict Nmin Nmax ηVec σj ),"jld")
 data_perf = Dict("NVec"=>NVec ,"ηVec" => ηVec,"RMSE" => RMSE,"RMSE_l" => RMSE_l)
 safesave(datadir("sims/LalaAbbott",name) ,data_perf)
