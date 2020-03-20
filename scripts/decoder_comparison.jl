using DrWatson
quickactivate(@__DIR__,"Random_coding")
using Distributions,JLD
include(srcdir("RM_model1D.jl"))
include(srcdir("decoders.jl"))
function Mcomparison(MVec,σVec,SNR,N;L=500,RN=1,ntrain=20,ntest=20,nNet=4,M=0,ηu=0,pn=0,circular=0)
    #Return RMSE matrix at different values of N and σ
    RMSE = zeros(length(MVec),length(σVec),nNet)
    RMSE_id = zeros(length(σVec),nNet)
    σ=1.; η = 1/SNR
    R =RN* N
    for net=1:nNet
            myn = Network(N,L,σ);renormalize!(myn,R);
            if circular==1
                myn.f1=vonMises
            end
            #v,r = myn(ntrial,η)
            for (j,σ) = enumerate(σVec)
                myn.σ = σ; myn.Z = ones(N); renormalize!(myn,R);
                RMSE_id[j,net]= evaluate_performance_ideal(myn,ntrain,η)
                for (i,M) = enumerate(MVec)
                    RMSE[i,j,net] = evaluate_performance_MLP(myn,ntrain,ntest,η,M=M)
                    println(M)
                end
                println(σ)
            end
    end
    return RMSE,RMSE_id
end

function decoder_comp(NVec,σVec,SNR;L=500,RN=1,ntrain=20,ntest=20,nNet=4,M=150,ηu=0,pn=0,circular=0)
    #Return RMSE matrix at different values of N and σ
    RMSE_id,RMSE_netMMSE,RMSE_MLP = [zeros(length(NVec),length(σVec),nNet) for i=1:3]
    for (i,N) = enumerate(NVec)
        σ=1.; η = 1/SNR
        R =RN* N
        for net=1:nNet
                myn = Network(N,L,σ);renormalize!(myn,R);
                #v,r = myn(ntrial,η)
                for (j,σ) = enumerate(σVec)
                    myn.σ = σ; myn.Z = ones(N); renormalize!(myn,R);
                    RMSE_id[i,j,net]= evaluate_performance_ideal(myn,ntrain,η)
                    RMSE_netMMSE[i,j,net]= evaluate_performance_netMMSE(myn,ntrain,η,M=M)
                    while isnan(RMSE_netMMSE[i,j,net])
                        myn = Network(N,L,σ);renormalize!(myn,R);
                        RMSE[i,j,net]= evaluate_performance_netMMSE(myn,ntrain,η,ηu=ηu,M=M,pn=pn)
                    end
                    RMSE_MLP[i,j,net]= evaluate_performance_MLP(myn,ntrain,ntest,η,M=M)
                end
        end
        println(N)
    end
    return RMSE_id,RMSE_netMMSE,RMSE_MLP
end

#Mcomparison
 MVec = [2,10,20,50,70,100,200,500];  σVec= 1.:3.:50.
 SNR=3.0; N= 80
RMSE,RMSE_id = Mcomparison(MVec,σVec,SNR,N,nNet=4)
dec="MLP"
name = savename("Mcomparison" , (@dict N SNR dec ),"jld")
data = Dict("MVec"=>MVec ,"σVec" => σVec,"RMSE" => RMSE,"RMSE_id" => RMSE_id)
safesave(datadir("sims/iidnoise",name) ,data)

#Different Decoders
# NVec= 10:10:120; σVec = 1.:3.:50.;SNR=3. ;M=150
# RMSE_id,RMSE_netMMSE,RMSE_MLP = decoder_comp(NVec,σVec,SNR,M=M,nNet=4)
# name = savename("deccomparison" , (@dict M SNR  ),"jld")
# data = Dict("MVec"=>NVec ,"σVec" => σVec,"RMSE_netMMSE" => RMSE_netMMSE,"RMSE_id" => RMSE_id,"RMSE_MLP" => RMSE_MLP)
# safesave(datadir("sims/iidnoise",name) ,data)
