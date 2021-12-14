##
using DrWatson
quickactivate(@__DIR__,"Random_coding")
#Strutcture and function for 1D model and ML-MSE inference
#using TensorBoardLogger,Logging
using Plots,LaTeXStrings
include(srcdir("network.jl"))
include(srcdir("decoder.jl"))
plotlyjs(size=(400,300))

##
x_min,x_max = -0.5,0.5
x_test1000 = range(x_min,x_max,length = 1000) 
x_test500 = range(x_min,x_max,length = 500) 
x_test = x_test1000
N,L = 30,500
σi = 5/L
η = 0.5
W = sqrt(1/L)*randn(N,L)
n = Network(W,σi,rxrnorm=0)
U,V = compute_tuning_curves(n,x_test)
R_train = hcat([V+sqrt(η)*randn(N,length(x_test)) for n=1:ntrain]...)
x_train = vcat([x_test for n=1:ntrain]...)
## Ideal decoder
ε_id  =  MSE_net_gon(n,η,MC=1,tol=1E-14)
data_args = DataArgs()
trn_args = TrainArgs(epochs=1)
data = GaussianDataset(V,η,x_test)
mlp_train_lin(data,trn_args)
##






function decodervsσ(η::AbstractFloat,W::AbstractArray,σVec::AbstractArray,
    Ml::Integer,Ms::Integer)
    N,L = size(W);
    dvsσ = Array{Dict}(undef,length(σVec));
    λ1 = sqrt(1/N)*randn(Ml,N)
    for (i,σi) = collect(enumerate( σVec))
        @info "σ = $σi"
        decs = Dict()
        n = Network(W,σi,rxrnorm=1)
        U,V = compute_tuning_curves(n,x_test)
        #Create noisy dataset
        R_train = hcat([V+sqrt(η)*randn(N,length(x_test)) for n=1:ntrain]...)
        x_train = vcat([x_test for n=1:ntrain]...)
        ## Ideal decoder
        ε_id  =  mse_ideal(n,η,x_test,R_train,x_train)
        decs[:ε_id] = ε_id
        ##Linear decoder
        λ_r,b_r = ridge_coefficients(V,x_test,k=length(x_test)*η)
        ε_r  = mean((x_train' -(λ_r'*R_train .+ b_r)).^2)
        decs[:ε_lin] = ε_r
        ##Random features decoder
        H_train = relu.(λ1*R_train)
        ε_rf,λ_rf,b_rf = mse_linear(H_train,x_train)
        decs[:ε_rf] =ε_rf
        ## Full train decoder - One hidden layer    
        mydec = Chain(Dense(N,Ml,relu),Dense(Ml,1,identity))
        data_trn = Flux.Data.DataLoader((R_train,x_train'),batchsize = 100)
        data_tst = data_trn
        mlp_decoder,history = train_mlp_decoder([data_trn,data_tst],mlp_decoder = 
    mydec ,η = 1E-4,epochs=100)
        decs[:ε_1l] = history[:mse_tst][end]
        decs[:history_1l] = history[:mse_tst]
        ## Full train decoder - Two hidden layer    
        mydec = Chain(Dense(N,Ms,relu),Dense(Ms,Ms,relu),Dense(Ms,1,identity))
        data_trn = Flux.Data.DataLoader((R_train,x_train'),batchsize = 100)
        data_tst = data_trn
        mlp_decoder,history = train_mlp_decoder([data_trn,data_tst],mlp_decoder = 
    mydec ,η = 1E-4,epochs=100)
        decs[:ε_2l] = history[:mse_tst][end]
        decs[:history_2l] = history[:mse_tst]
        dvsσ[i] = decs
    end
    return dvsσ
end
##
#Initialize parameters
x_min,x_max = -0.5,0.5
x_test1000 = range(x_min,x_max,length = 1000) 
x_test500 = range(x_min,x_max,length = 500) 
x_test = x_test500
N,L = 30,500
σi = 30/L
η = 0.5
Ml,Ms = 600,100
ntrain = 50
#encoding and decoding matrices
W = sqrt(1/L)*randn(N,L)

σVec = (5:10:55)/L
f = relu
fname = string(Symbol(f))
##
dvsσ = decodervsσ(η,W,σVec,Ml,Ms)
##
name = savename("decoders_relu" , (@dict Ml Ms η),"jld2")
data = Dict("σVec" => σVec ,"W" => W,"dvsσ" => dvsσ)
safesave(datadir("sims/iidnoise/MLPdec",name) ,data)
ε_idVec = [dvsσ[i][:ε_id] for i = 1:length(σVec)]
ε_linVec = [dvsσ[i][:ε_lin] for i = 1:length(σVec)]
ε_rfVec = [dvsσ[i][:ε_rf] for i = 1:length(σVec)]
ε_1lVec = [dvsσ[i][:ε_1l] for i = 1:length(σVec)]
ε_2lVec = [dvsσ[i][:ε_2l] for i = 1:length(σVec)]
##
η= 0.3; σi = 25/L
n = Network(W,σi,rxrnorm=1)
U,V = compute_tuning_curves(n,x_test)
λ1 = sqrt(1/N)*randn(Ml,N)
#Create noisy dataset
R_train = hcat([V+sqrt(η)*randn(N,length(x_test)) for n=1:ntrain]...)
x_train = vcat([x_test for n=1:ntrain]...)
##Random features decoder
H_train = relu.(λ1*R_train)
ε_rf,λ_rf,b_rf = mse_linear(H_train,x_train)
#Full train decoder - One hidden layer    
mydec1l = Chain(Dense(N,Ml,relu),Dense(Ml,1,identity))
data_trn = Flux.Data.DataLoader((R_train,x_train'),batchsize = 100)
data_tst = data_trn
mlp_decoder1l,history1l = train_mlp_decoder([data_trn,data_tst],mlp_decoder = 
    mydec1l ,η = 1E-4,epochs=200)
## Full trained decoder 2 hidden layer
mydec2l = Chain(Dense(N,Ms,relu),Dense(Ms,Ms,relu),Dense(Ms,1,identity))
data_trn = Flux.Data.DataLoader((R_train,x_train'),batchsize = 100)
data_tst = data_trn
mlp_decoder2l,history2l = train_mlp_decoder([data_trn,data_tst],mlp_decoder = 
    mydec2l ,η = 1E-4,epochs=200)
ε_id  =  mse_ideal(n,η,x_test,R_train,x_train)

