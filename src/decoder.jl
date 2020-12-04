using DrWatson
@quickactivate "Random_coding"
using Flux
include(srcdir("network.jl"))

##Functions for more advanced decoders


function supervised_dataset(n::Network,η::Float64; ntrain=20000,ntest = 5000,bsize=50)
    U,V = compute_tuning_curves(n,x_test);
    N,L = size(V)
    x_trn,x_tst = rand(x_test,ntrain),rand(x_test,ntest);
    x_trn = Flux.onehotbatch(x_trn,x_test)
    x_test = Flux.onehotbatch(x_tst,x_test)
    r_trn=  hcat([V[:,x_trn[:,i]] .+ sqrt(η)*randn(N)  for i = 1:ntrain]...)
    hcat([rand.(p_r_x(x,θ)) for x=x_tst]...);
    data_trn = Flux.Data.DataLoader((r_trn,Flux.onehotbatch(x_trn,x_test)),batchsize = bsize);
    data_tst = Flux.Data.DataLoader((r_tst,Flux.onehotbatch(x_tst,x_test)),batchsize = bsize);
    return data_trn,data_tst
end
