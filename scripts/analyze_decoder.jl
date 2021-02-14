using DrWatson
quickactivate(@__DIR__,"Random_coding")
using JLD2
#Strutcture and function for 1D model and ML-MSE inference
include(srcdir("network.jl"))
include(srcdir("decoder.jl"))

function Mcomparison(data,MVec::AbstractArray;nepochs=100,ηl=1e-3)
    data_trn,data_tst = [data...]
    mlp_decoders= []
    for M = MVec
        @info "M = $M"
        d = Dict()
        #my_dec = Chain(Dense(N,M,sigmoid),Dense(M,1,linear))
        d[:dec],d[:hist] = train_mlp_decoder((data_trn,data_tst),epochs = nepochs,M=M,η =ηl);
        push!(mlp_decoders,d)
    end
    return mlp_decoders
end

function analyze_decoder(η::AbstractFloat,W::AbstractArray,σVec::AbstractArray,MVec::AbstractArray)
    #Find optimal σ for  agiven network with the noise level η
    N,L = size(W);
    dvsσ = Array{Dict}(undef,length(σVec));
    Threads.@threads for (i,σi) = collect(enumerate( σVec))
        @info "σ = $σi"
        n = Network(N,L,σi,rxrnorm=0);
        n.W = W
        decoders = Dict()
        data_trn,data_tst = iidgaussian_dataset(n,η,onehot=true);
        decoders[:ideal_decoder] = ideal_decoder_iidgaussian(n,η,x_test)
         ~,decoders[:mse_id] = ideal_loss(n,η,x_test,data_tst);
        decoders[:prob_decoder],decoders[:history_prob] = train_prob_decoder((data_trn,data_tst),epochs=100);
        decoders[:mlp_decoders] = Mcomparison((data_trn,data_tst),MVec)
        dvsσ[i] = decoders
    end
    return dvsσ
end

N,L =35,500;
NVec = 30:5:80
η = 0.5
σVec = (5:3:50)/L;
MVec = Int.(round.(10 .^range(log10(2),log10(500),length=15)))[3:12]
dvsσVec = []
for N=NVec
    W = sqrt(1/L)*randn(N,L);
    dvsσ = analyze_decoder(η,W,σVec,MVec)
    push!(dvsσVec,dvsσ)
end
Nmin,Nmax = first(Nmin,Nmax)
name = savename("relu_lin_decoder" , (@dict Nmin Nmax  L η),"jld2")
data = Dict("σVec" => σVec ,"MVec" => MVec, "W" => W,"dvsσ" => dvsσ)
safesave(datadir("sims/iidnoise/MLPdec",name) ,data)
