using DrWatson
quickactivate(@__DIR__,"Random_coding")
using Distributions,JLD
include(srcdir("RM_model1D.jl"))
include(srcdir("decoders.jl"))
#Define distribution of widths
function σdist(NVec,SNRVec,σd,fdec::Function;L=500,η=1.0,ntrial=30,nNet=4,M=0,ηu=0,pn=0)
    RMSE = zeros(length(NVec),length(SNRVec),nNet)
    for (i,N) = enumerate(NVec)
        R = N;
        for (j,SNR)=enumerate(SNRVec)
            η = 1/SNR;
            for net=1:nNet
                myn = Network(N,L,rand(σd,L));renormalize!(myn,R);
                RMSE[i,j,net]= fdec(myn,ntrial,η,ηu=ηu,M=M,pn=pn)
                while isnan(RMSE[i,j,net])
                    myn = Network(N,L,σ);renormalize!(myn,R);
                    RMSE[i,j,net]= fdec(myn,ntrial,η,ηu=ηu,M=M,pn=pn)
                end
            end
        end
        println("N=",N)
    end
    return RMSE
end


SNRVec = 10 .^(-0.3:0.2:1.1); NVec = Int.(round.(10 .^(0.9:0.1:2.3)));Nmin,Nmax = first(NVec),last(NVec);
a=0.01;σmax=50.;σmin=1. ; θ = (-σmax^(-a)+σmin^(-a))^(1/a)
σd = Truncated(Pareto(a,θ),σmin,σmax);dist = "Pareto0.01"
fdec = evaluate_performance_ideal
RMSE=σdist(NVec,SNRVec,σd,fdec)
name = savename("Nsigmadist" , (@dict Nmin Nmax dist  ),"jld")
data = Dict("NVec"=>NVec ,"SNRVec" => SNRVec,"RMSE" => RMSE,"σd" => σd)
safesave(datadir("sims/sigmadist",name) ,data)
