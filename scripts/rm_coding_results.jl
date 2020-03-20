using DrWatson
quickactivate(@__DIR__,"Random_coding")
using Distributions,JLD
include(srcdir("RM_model1D.jl"))
include(srcdir("decoders.jl"))
function Nvsσ(NVec,σVec,SNR,fdec::Function;L=500,RN=1,ntrial=30,nNet=4,M=0,ηu=0,pn=0,circular=0)
    #Return RMSE matrix at different values of N and σ
    RMSE = zeros(length(NVec),length(σVec),nNet)
    σ=1.
    for (i,N) = enumerate(NVec)
        η = 1/SNR
        R =RN* N
        Threads.@threads for net=1:nNet
            myn = Network(N,L,σ);renormalize!(myn,R);
            if circular==1
                myn.f1=vonMises
            end
            #v,r = myn(ntrial,η)
            for (j,σ) = enumerate(σVec)
                myn.σ = σ; myn.Z = ones(N); renormalize!(myn,R);
                RMSE[i,j,net]= evaluate_performance_ideal(myn,ntrial,η,ηu=ηu,pn=pn,circular=circular)
                while isnan(RMSE[i,j,net])
                    myn = Network(N,L,σ);renormalize!(myn,R);
                    RMSE[i,j,net]= fdec(myn,ntrial,η,ηu=ηu,M=M,pn=pn)
                end
                println(N,"  ",σ)
            end
        end
        println("N=",N)
    end
    return RMSE
end

function NvsSNR(NVec,SNRVec,fdec::Function;L=500,ntrial=30,nNet=4,σVec=1.:1.:50.)
    #return RMSE opt in the N-SNR pairs
    RMSE_opt_SNR,σ_opt_SNR = [zeros(length(NVec),length(SNRVec)) for n=1:2]
    for (j,SNR) = enumerate(SNRVec)
        RMSE = Nvsσ(NVec,σVec,SNR,fdec)
        RMSE_m = mean(RMSE,dims=3)[:,:]; RMSE_v = var(RMSE,dims=3)[:,:];
        RMSE_opt,σoptpos_id = findmin(RMSE,dims=2)
        σ_opt = [σVec[pos[2]] for pos=σoptpos_id];σ_opt_var = var(σ_opt,dims=3)[:];
        σ_opt = mean(σ_opt,dims=3)[:];RMSE_opt = mean(RMSE_opt,dims=3)[:]
        RMSE_opt_SNR[:,j] = RMSE_opt;σ_opt_SNR[:,j] = σ_opt;
        println(SNR)
    end
    return RMSE_opt_SNR,σ_opt_SNR
end

#Performance for a fixed decoder
# L=500; NVec = 10:5:100;σVec = 1.:1.:50.;SNR=1;c=1
# dec="ideal"
# decodern = "evaluate_performance_"*dec
# decoder=getfield(Main,Symbol(decodern))
# RMSE= Nvsσ(NVec,σVec,SNR,decoder,nNet=8,circular=c)
# Nmin,Nmax = first(NVec),last(NVec);
# println(dec)
# name = savename("Nsigma" , (@dict Nmin Nmax SNR dec c ),"jld")
# data = Dict("NVec"=>NVec ,"σVec" => σVec,"RMSE" => RMSE)
# safesave(datadir("sims/iidnoise",name) ,data)

#NvsSNR heatmap
NVec = 10:10:150;σVec = 1.:1.:50.; SNRVec = [0.3,0.5,0.8,1.,1.3,1.5,1.7,2.,3.,5.]
dec="ideal";decodern = "evaluate_performance_"*dec
decoder=getfield(Main,Symbol(decodern))
RMSE_opt,σ_opt = NvsSNR(NVec,SNRVec,decoder)
Nmin,Nmax = first(NVec),last(NVec);SNRmin,SNRmax = first(SNRVec),last(SNRVec);
name = savename("NvsSNR" , (@dict Nmin Nmax SNRmin SNRmax dec ),"jld")
 data = Dict("NVec"=>NVec ,"SNRVec" => SNRVec,"RMSE_opt" => RMSE_opt,"σ" => σ_opt)
  safesave(datadir("sims/iidnoise",name) ,data)

#Cycle to obtain Nvssigma for different decoders/noise setting

#Poisson Noise
# NVec = 10:5:80; σVec = 1.:1.:50.; SNR=5.0;Nmin,Nmax = first(NVec),last(NVec);
# ηu=0;pn=1
# dec = "ideal"
# decodern = "evaluate_performance_"*dec;   decoder=getfield(Main,Symbol(decodern))
# RMSE= Nvsσ(NVec,σVec,SNR,decoder,nNet=4,ηu=ηu,ntrial=30,pn=pn)
# name = savename("Nsigma" , (@dict Nmin Nmax SNR dec pn ),"jld")
# data = Dict("NVec"=>NVec ,"σVec" => σVec,"RMSE" => RMSE)
# safesave(datadir("sims/pn",name) ,data)

# #Input Noise
# NVec = 20:5:100; σVec = 1.:1.:50.; SNR=1.0;Nmin,Nmax = first(NVec),last(NVec);
# dec = "ideal"
# decodern = "evaluate_performance_"*dec;   decoder=getfield(Main,Symbol(decodern))
# L=500;ηu = 1/L
# RMSE= Nvsσ(NVec,σVec,SNR,decoder,nNet=4,ηu=ηu,L=L,ntrial=50)
# name = savename("Nsigma" , (@dict Nmin Nmax SNR dec ηu*L ),"jld")
# data = Dict("NVec"=>NVec ,"σVec" => σVec,"RMSE" => RMSE)
# safesave(datadir("sims",name) ,data)
#SNRVec = 10 .^(-1:0.2:1); NVec = Int.(round.(10 .^(0.9:0.1:2.3)))

# NVec = 10:10:120; σVec = 1.:1.:50.; SNR=2.0;Nmin,Nmax = first(NVec),last(NVec); M=150
# #Different decoders
# for dec = ["MLP","netMMSE"]
#     decodern = "evaluate_performance_"*dec
#     decoder=getfield(Main,Symbol(decodern))
#     RMSE= Nvsσ(NVec,σVec,SNR,decoder,nNet=4,M=M)
#     Nmin,Nmax = first(NVec),last(NVec);
#     println(dec)
#     name = savename("Nsigma" , (@dict Nmin Nmax SNR dec ),"jld")
#     data = Dict("NVec"=>NVec ,"σVec" => σVec,"RMSE" => RMSE)
#     safesave(datadir("sims/iidnoise",name) ,data)
# end
