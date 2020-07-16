using DrWatson
quickactivate(@__DIR__,"Random_coding")
#Strutcture and function for 1D model and ML-MSE inference
using Distributions,StatsBase , LinearAlgebra, MultivariateStats,Random,SparseArrays,Flux,ProgressMeter
using TensorBoardLogger,Logging
include(srcdir("network.jl"))
function decoder_comparison(myn::Network,decoder)
    b = sum(V.^2,dims=1)'
    H = exp.((V'*Rtst .-0.5*b )/(η));Zh = sum(H,dims=1);H = H./Zh
    p_x_r = decoder(Rtst)
end
function train!()
    for n=1:n_epochs
    @info "Epoch $(n)"
    p = Progress(length(data))
    for (r,xt) = data
        l, back = Flux.pullback(ps) do
            loss(r,xt)
        end
        grad = back(1f0)
        Flux.Optimise.update!(opt,ps,grad);
        next!(p; showvalues=[(:loss, l)])
    end
    #losstot = β_loss(encoder,decoder,U,x_test')
    push!(loss_tst,loss(Rtst,Flux.onehotbatch(x_tst',x_test)))
end
end

decoder = Chain(Dense(N,M),softmax);
#Crossentropy loss
loss(r,x_true) = Flux.crossentropy(decoder(r),x_true)
