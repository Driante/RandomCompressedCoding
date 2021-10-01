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
function decodervsσ(η::AbstractFloat,W::AbstractArray,σVec::AbstractArray,Ml,Ms)
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
η = 0.3
Ml,Ms = 600,100
ntrain = 50
#encoding and decoding matrices
W = sqrt(1/L)*randn(N,L)

σVec = (5:10:55)/L
dvsσ = decodervsσ(η,W,σVec,Ml,Ms)
name = savename("decoders_relu" , (@dict Ml Ms η),"jld2")
data = Dict("σVec" => σVec ,"W" => W,"dvsσ" => dvsσ)
safesave(datadir("sims/iidnoise/MLPdec",name) ,data)
ε_idVec = [dvsσ[i][:ε_id] for i = 1:length(σVec)]
ε_linVec = [dvsσ[i][:ε_lin] for i = 1:length(σVec)]
ε_rfVec = [dvsσ[i][:ε_rf] for i = 1:length(σVec)]
ε_1lVec = [dvsσ[i][:ε_1l] for i = 1:length(σVec)]
ε_2lVec = [dvsσ[i][:ε_2l] for i = 1:length(σVec)]
plot(σVec,)
##
ε_rfVec,ε_mlpVec = [[] for n=1:2]
ε_nfVec,ε_nVec,ε_rVec,J_nfVec,J_nVec,J_rVec,ε_idVec = [[] for n=1:7]
for σi = σVec
    n = Network(W,σi,rxrnorm=1)
    U,V = compute_tuning_curves(n,x_test)
    #Create noisy dataset
    R_train = hcat([V+sqrt(η)*randn(N,length(x_test)) for n=1:ntrain]...)
    x_train = vcat([x_test for n=1:ntrain]...)

    #Linear regression
    #ε_nf,λ_nf,b_nf = mse_linear(V,x_test)
    λ_r,b_r = ridge_coefficients(V,x_test,k=length(x_test)*η)
    ε_r  = mean((x_train' -(λ_r'*R_train .+ b_r)).^2)

    #ε_nf = mean((x_test' - (λ_r'*V .+ b_r)).^2)
    push!(ε_rVec,ε_r)
    #Random features
    H_nf = relu.(λ1*V)
    H_train = relu.(λ1*R_train)
    ε_rf,λ_rf,b_rf = mse_linear(H_train,x_train)
    push!(ε_rfVec,ε_rf)

    # MLP regression
    mydec = Chain(Dense(N,M,relu),Dense(M,1,identity))
    data_trn = Flux.Data.DataLoader((R_train,x_train'),batchsize = 100)
    data_tst = data_trn
    mlp_decoder,history = train_mlp_decoder([data_trn,data_tst],mlp_decoder = 
    mydec ,η = 1E-4,epochs=100)
    ε_mlp = history[:mse_tst][end]

    push!(ε_mlpVec,ε_mlp)

    #ε_nf,λ_nf,b_nf = mse_linear(V,x_test)
    #λ_r,b_r = ridge_coefficients(V,x_test,k=length(x_test)*η)
    #ε_r  = mean((x_train' -(λ_r'*R_train .+ b_r)).^2)
    #ε_nf = mean((x_test' - (λ_r'*V .+ b_r)).^2)
    #ε_id  =  mse_ideal(n,η,x_test,R_train,x_train)
    #push!(ε_nfVec,ε_nf)
    #push!(ε_nVec,ε_n)
    #push!(ε_rVec,ε_r)
    #push!(J_nfVec,sum(λ_nf.^2))
    #push!(J_nVec,sum(λ_n.^2))
    #push!(J_rVec,sum(λ_r.^2))
    #push!(ε_idVec,ε_id)
end
#plot(σVec,ε_nVec,yaxis=:log10,m=:o)


plot!(σVec,ε_rfVec,yaxis=:log10,m=:o)
plot!(σVec,ε_mlpVec,yaxis=:log10,m=:o)
#plot!(σVec,ε_nfVec,yaxis=:log10,m=:o)
#plot!(σVec,ε_idVec,yaxis=:log10,m=:o)
#plot!(σVec,ε_nfVec + η*J_rVec,yaxis=:log10,m=:o)


##

