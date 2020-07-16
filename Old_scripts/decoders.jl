using DrWatson
quickactivate(@__DIR__,"Random_coding")
using Distributions,JLD,MultivariateStats,Flux
include(srcdir("RM_model1D.jl"))
function dataset(r;bsize=50)
    N,ntrain,L = size(r)
    x = Float32.(reshape(r,N,:)); y = Float32.(vcat([i*ones(ntrain).-L/2 for i=1:L]...))
    dtrn=[]; prmindex = shuffle(1:(size(x)[2]));
    x = x[:,prmindex]; y= y[prmindex]
    i=1; bsize=50
    while i < (size(x)[2] - bsize)
        push!(dtrn,(x[:,i:i+bsize-1],y[i:i+bsize-1]))
        i += bsize
    end
    return dtrn
end
function evaluate_performance_MLP(n::Network,ntrain,ntest,η;bsize=50,M=100,pn=0,ηu=0,nrep=100)
    N,L = size(n.W)
    v,rtrn = n(ntrain,η);  v,rtst = n(ntest,η);
    xtst = reshape(rtst,N,:); ytst = vcat([i*ones(Int.(ntest)).-L/2 for i=1:L]...)
    dtrn = dataset(rtrn)
    loss(x,y) = Flux.mse(m(x)',y)
    m = Chain(Dense(N,M,σ),Dense(M,1))
    opt = ADAM(0.01); evalcb = () -> @show(sqrt.(loss(xtst,ytst)))
    Flux.train!(loss,Flux.params(m),Flux.repeat(dtrn,nrep),opt,cb=Flux.throttle(evalcb,10))
    errors = reshape(ytst-vec(m(xtst).data),ntest,:)
    RMSE= sqrt(mean(errors.^2))
    return RMSE
end
