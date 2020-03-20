using DrWatson
quickactivate(@__DIR__,"Random_coding")
using Distributions,StatsBase , LinearAlgebra, MultivariateStats,Random,Knet#Ideal Decoder
using Knet:Data
using Base.Iterators: flatten
using IterTools: takenth
#REDO in FLUX
struct Chain
    layers
    Chain(layers...) = new(layers)
end
#When called with an arugment, it apply the function layer to every element, recursively modify x
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)
struct Layer1; w; b; f; end
Layer1(i::Int,o::Int,f=relu) = Layer1(param(o,i,atype=Knet.Array{Float64}),param0(o;atype=Knet.Array{Float64}),f)
(l::Layer1)(x) = l.f.(l.w * x .+ l.b)

MSE(y_pred,y) = mean((y_pred-y).^2)
(c::Chain)(x,y)= MSE(vec(c(x)),y)
(c::Chain)(d::Data) = mean(c(x,y) for (x,y) in d)

function split_data(r::Array{Float64,3},bsize)
    N,ntrial,L = size(r);
    x = reshape(r[:,1:Int(ntrial/2),:],N,:); y = vcat([i*ones(Int.(ntrial/2)).-L/2 for i=1:L]...)
    dtrn = minibatch(x,y,bsize,shuffle= true)
    x = reshape(r[:,Int(ntrial/2+1):end,:],N,:); y = vcat([i*ones(Int.(ntrial/2)).-L/2 for i=1:L]...)
    dtst = minibatch(x,y,bsize,shuffle= true)
    return dtrn,dtst
end
#Performance evaluation
function evaluate_performance_MLP(n::Network,ntrial::Integer,η::Float64;bsize=100,M=100)
    #Evaluate mean square error for a given network at a given level of noise, computing
    #RMSE
    N,L = size(n.W)
    v,r = n(ntrial,η);  dtrn,dtst = split_data(r,bsize)
    model=Chain(Layer1(N,M,relu),Layer1(M,1,identity))
    progress!(adam(model, repeat(dtrn,100),lr=0.01))
    @show  sqrt.(model(dtst))
    y = vcat([i*ones(Int.(ntrial)).-L/2 for i=1:L]...)
    errors = reshape(ytst-vec(m(xtst).data),ntest,:)
    RMSE= mean(vec(sqrt.(mean(errors.^2,dims=1))))
    return RMSE
end
