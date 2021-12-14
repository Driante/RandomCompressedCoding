using DrWatson
@quickactivate "Random_coding"
using Flux
using Parameters: @with_kw
using ProgressMeter: Progress, next!
##
@with_kw mutable struct DataArgs
    ntrn = 50        #Number of noisy samples for the same stimulus x train
    ntst = 50         #Number of noisy samples for the same stimulus x test
    mb = 100           #Size of minibatches
    shuff = true      #Shuffle data
end

@with_kw mutable struct TrainArgs
    lr = 1e-3              # learning rate
    epochs = 50             # number of epochs
    M = 500                 # latent dimension
    verbose_freq = 10       # logging for every verbose_freq iterations
    opt = ADAM             #Optimizer
    f = relu               ##Specify non linearity
end
##
function GaussianDataset(V::AbstractMatrix,η::AbstractFloat,
        x_test::AbstractArray; kws...)
    #Input: matrix of tuning curves in the shape N x L
    N,L = size(V)
    data_args = DataArgs(;kws...)
    R_trn = hcat([V .+ sqrt(η)*randn(N,L) for n=1:data_args.ntrn]...)
    x_trn = hcat([x_test' for n=1:data_args.ntrn]...)
    @info size(x_trn),size(R_trn)
    R_tst = hcat([V .+ sqrt(η)*randn(N,L) for n=1:data_args.ntst]...)
    x_tst = hcat([x_test' for n=1:data_args.ntst]...)
    data_trn = Flux.Data.DataLoader((R_trn,x_trn),batchsize = data_args.mb,shuffle = data_args.shuff);
    data_tst = Flux.Data.DataLoader((R_tst,x_tst));
    return data_trn,data_tst
end

## Linear Loss
mse_loss(dec,r,x) = Flux.mse(dec(r) , x)

function mlp_train_lin(data,mlp_decoder = nothing,kws...)
    #Train MLP decoder on linear data
    data_trn,data_tst = data[1],data[2]
    N,mb = size(first(data_trn)[1])
    trn_args = TrainArgs(; kws...)
    if isnothing(mlp_decoder)
        #If decoder is not initialized, single layer NN
        mlp_decoder = Chain(Dense(N,trn_args.M,trn_args.f),
            Dense(trn_args.M,1,identity))     #Decoder is a MLP with hidden layer of size M
    end
    #Select optimizer and parameters to train
    opt = trn_args.opt(trn_args.lr)
    ps = Flux.params(mlp_decoder)
    #Save training history
    history = Dict(:mse_trn => Float32[],:mse_tst => Float32[])
    for epoch = 1:trn_args.epochs
        @info "Epoch $(epoch)"
        progress = Progress(length(data_trn))
        loss = 0
        for d in data_trn
            l, back = Flux.pullback(ps) do
                mse_loss(mlp_decoder,d...)
            end
            grad = back(1f0)
            Flux.Optimise.update!(opt, ps, grad)
            next!(progress; showvalues=[(:loss, l),(:epoch,epoch)])
            loss += l
        end
        loss /= length(data_trn)
        #Update history at the end of each epoch iteration
        push!(history[:mse_trn],loss)
        push!(history[:mse_tst],mean([mse_loss2(mlp_decoder,dtt...) 
            for dtt in data_tst]))
    end
    return mlp_decoder,history
end


##
#Linear- Generalized Linear decoders
function ridge_coefficients(V,x;k = 1E-200)
    λ = (V*V' + k*I)\(V*x)
    b = λ'*mean(V,dims=2) .- mean(x)
    return λ, b
end
function mse_linear(V,x; k = 1E-200)
    λ,b = ridge_coefficients(V,x,k=k)
    x_ext = λ'V .+ b
    ε2 = mean((x_ext -x').^2)
    return ε2,λ,b
end
function mse_ideal(V,η,x_test,R_t,x_t)
    #Compute losses (cross entropy and mse) on a dataset
    λ_id = V'/η;
    b_id = -sum((V').^2,dims=2)/(2*η)
    H = Flux.softmax(λ_id*R_t .+ b_id)
    x_ext = x_test'*H
    ε2 = mean((x_ext - x_t').^2)
    return ε2
end