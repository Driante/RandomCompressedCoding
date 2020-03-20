using DrWatson
quickactivate(@__DIR__,"Random_coding")
using Distributions,JLD,Plots,Flux,Zygote
import Zygote: Params, gradient
include(srcdir("RM_model1D.jl"))
include(srcdir("decoders.jl"))
ϵ =1E-9
function predict_x(U)
    V = W*U
    R = V + sqrt(η)*randn(N,L)
    h = exp.((R'*V)/η);Z = sum(h,dims=2)
    w = 1:L
    x_ext = (h./Z)*w
    return x_ext
end
function MSE(x,U)
    x_ext = predict_x(U)
    mse = mean((x.-x_ext).^2)
    return mse
end
λ=10
loss(U) = mse(x,U) + λ*(mean(std(W*U,dims=2))-1)
function MSE(x,U,ntrial)
    mse = 0
    for n=1:ntrial
        mse += MSE(x,U)
    end
    mse /= ntrial
    return mse
end

function apply!(o::ADAM, x, Δ)
  η, β = o.eta, o.beta
  mt, vt, βp = get!(o.state, x, (zero(x), zero(x), β))
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ^2
  @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵ) * η
  o.state[x] = (mt, vt, βp .* β)
  return Δ
end
function update!(x::AbstractArray, x̄)
  x .+= x̄
  return x
end

function update!(opt, x, x̄)
  x .-= apply!(opt, x, x̄)
end

function update!(opt, xs::Params, gs)
  for x in xs
    gs[x] == nothing && continue
    update!(opt, x, gs[x])
  end
end
L=500; ntrial=20;ηu=0.5/L
N=50;σ = 5.;SNR=1.; R= N; η = .4
myn = Network(N,L,σ);
U = hcat([myn.f1(x,σ,L) for x=1:L]...)
W=randn(N,L)
x_ext = predict_x(U)
x=1:L;
d = (x,U)
o = ADAM()
ps=Params([W])
l = zeros(100)
for n=1:100
    gs = gradient(ps) do
        MSE(d...)
      end
    update!(o,ps,gs)
    l[n] = MSE(x,U,ntrial)
    println(MSE(x,U,ntrial))
end

function loss2(U)
    V=W*U
    return mean(var(V,dims=2))
end
g =gradient(() -> loss2(U),Params([W]))

R=N
myn = Network(N,L,σ); renormalize!(myn,R);
v,r = myn(ntrial,η);
evaluate_performance_ideal(myn,ntrial,η)
