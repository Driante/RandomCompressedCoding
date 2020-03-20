using DrWatson
quickactivate(@__DIR__,"Random_coding")
#Strutcture and function for 1D model and ML-MSE inference
using Distributions,StatsBase , LinearAlgebra, MultivariateStats,Random,SparseArrays

mutable struct Network2l; W::Array{Float64,2};σ;U;V;x_test;Z;end
function Network2l(N,L,σ;ntest=500)
    W = sqrt(1/L)*randn(N,L)
    centers = 1/L:1/L:1; x_test =range(0,length=ntest,stop=1)
    #tuning curves of first layer neurons
    U = zeros(L,ntest)
    for (j,c_j) = enumerate(1/L:1/L:1)
        U[j,:]=exp.((1.0/(2*π*σ/ntest)^2)*(cos.((c_j.-x_test)*2π).-1))
    end
    U ./= sqrt.(sum(U.^2*1/ntest,dims=2))
    V = W*U; Z = sqrt.(var(V,dims=2)) ; V ./= Z
    return Network2l(W,σ,U,V,x_test,Z)
end
function generate_noisy_responses(n::Network2l,η,η_u,ntrial)
    N,L = size(n.W); ntest = length(n.x_test)
    R = zeros(N,ntest,ntrial)
    for i=1:ntrial
        R[:,:,i]= n.V + sqrt(η)*randn(N,ntest) + sqrt(η_u)*n.W*randn(L,ntest)./n.Z
    end
    return R
end
function decode(r,V,iΣ,x_test)
    logl = vec(-sum((r.-V).*(iΣ*(r.-V)),dims=1)*0.5)
    likelihood = exp.(logl); likelihood ./=sum(likelihood)
    S = sum(sin.(2π*(x_test)).*likelihood); C = sum(cos.(2π*(x_test)).*likelihood)
    k_MSE = mod(atan(S,C)/(2π),1)
    return k_MSE
end
function evaluate_performance(n::Network2l,η,η_u,ntrial)
    W = n.W; Z = n.Z ; N,L = size(W);  ntest = length(n.x_test)
    R = generate_noisy_responses(n,η,η_u,ntrial)
    errors = zeros(ntrial,ntest)
    Σ = (η_u*W*W')./(Z*Z') + η*I; iΣ = inv(Σ)
    for t = 1:ntrial
        for (i,x) = enumerate(n.x_test)
            x_ext = decode(R[:,i,t],n.V,iΣ,n.x_test)
            errors[t,i] = x_ext - x
        end
    end
    errors = abs.(errors)
    errors[errors.>1/2] .-=1;
    RMSE= sqrt(mean(errors.^2))
    return RMSE
end

N = 15; L = 500;  σ = 1.;
η=0.5; η_u = 0.1;
ntrial=20
α = 0.1 ;
Li = 500;NVec = vcat(40:2:60);σVec = 1.:3.:30.; η_uVec = 0.1:0.2:1.
C = 15+ α*Li
RMSE = zeros(length(NVec),length(σVec),length(η_uVec))
for (k,η_u) = enumerate(η_uVec)
    for (i,N) = enumerate(NVec)
        L = L = Int(round((C - N)/α))
        println(L," ",N," ",η, " ",η_u)
        for n=1:6
            for (j,σ) = enumerate(σVec)
                    myn = Network2l(N,L,σ);
                    RMSE[i,j,k] +=  evaluate_performance(myn,η,η_u,ntrial)/4
                    println(RMSE[i,j,k])
            end
        end
    end
end
name = savename("NvsL" , (@dict η α Li  ),"jld")
data = Dict("NVec"=>NVec ,"σVec" => σVec,"η_uVec" => η_uVec,"RMSE" => RMSE,)
safesave(datadir("sims",name) ,data)
