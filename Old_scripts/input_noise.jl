using DrWatson
quickactivate(@__DIR__,"Random_coding")
using Distributions,JLD
include(srcdir("RM_model1D.jl"))
function evaluate_comparisoncvsi_ideal(n::Network,ntrial::Integer,η::Float64,ηu;pn=0,M=0,circular=0)
    #Compare the same network performance using idependent vs correlated noise
    N,L = size(n.W);
    if n.f1==vonMises
        circular=1
    end
    errors,errorsi = [zeros(ntrial,L) for i=1:2];
    #compute errors in correlated case
    v,r = n(ntrial,η,ηu=ηu,pn=pn);
    Σ = η*Matrix(I,N,N) + (ηu*n.W*n.W')/(n.Z^2);iΣ = inv(Σ)
    for x=1:L
        for t=1:ntrial
            errors[t,x] = x - ideal_decoder(r[:,t,x],v,η,iΣ=iΣ,pn=pn,circular=circular)
        end
    end
    errors = abs.(errors)
    if circular ==1
        errors[errors.>L/2] .-=L
    end
    #compute errors in case where correlations goes to 0 but we mantain diagonal term
    Σi = Diagonal(Σ);iΣi = inv(Σi)
    v,r = n(ntrial,diag(Σi),ηu=0,pn=pn);
    for x=1:L
        for t=1:ntrial
            errorsi[t,x] = x - ideal_decoder(r[:,t,x],v,η,iΣ=iΣi,pn=pn,circular=circular)
        end
    end
    errorsi = abs.(errorsi)
    if circular ==1
        errorsi[errorsi.>L/2] .-=L
    end
    RMSE= mean(vec(sqrt.(mean(errors.^2,dims=1))))
    RMSEi= mean(vec(sqrt.(mean(errorsi.^2,dims=1))))
    return  RMSE,RMSEi#,maximum(abs.(errors))
end

function Nvsσcvsi(NVec,σVec,SNR,ηu;L=500,RN=1,ntrial=30,nNet=4,M=0,pn=0)
    RMSE,RMSEi = [zeros(length(NVec),length(σVec),nNet) for n=1:2]
    for (i,N) = enumerate(NVec)
        η = 1/SNR
        R =RN* N
        σ=1.
        for net=1:nNet
            myn = Network(N,L,σ,f1=vonMises);renormalize!(myn,R);
            #v,r = myn(ntrial,η)
            for (j,σ) = enumerate(σVec)
                myn.σ = σ; myn.Z = ones(N); renormalize!(myn,R);
                RMSE[i,j,net],RMSEi[i,j,net]= evaluate_comparisoncvsi_ideal(myn,ntrial,η,ηu,M=M,pn=pn)
            end
        end
        println("N=",N)
    end
    return RMSE,RMSEi
end

L=500; NVec = 10:5:100;σVec = 1.:2.:50.;
ηu =0.5/L;ntrial=100;
 SNR=3;
#RMSEo,σo,RMSEoi,σoi = [zeros(length(NVec),length(ηuVec)) for n=1:4];
RMSE,RMSEi = Nvsσcvsi(NVec,σVec,SNR,ηu,ntrial=ntrial)
Nmin,Nmax = first(NVec),last(NVec);
name = savename("cvsi", (@dict Nmin Nmax SNR ηu*L ),"jld")
data = Dict("NVec"=>NVec ,"σVec"=> σVec,"RMSE" => RMSE, "RMSEi" => RMSEi)
safesave(datadir("sims/inputnoise",name) ,data)
