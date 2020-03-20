using DrWatson
quickactivate(@__DIR__,"Random_coding")
#Strutcture and function for 1D model and ML-MSE inference
using Distributions,StatsBase , LinearAlgebra, MultivariateStats,Random,SparseArrays

#Structure which specify the connectivity matrix, the width of the first layer tuning curves, and the tuning function of the two layers
mutable struct Network; W::Array{Float64,2};σ;f1::Function; f2::Function;Z;ϕ;end
#First layer vector response
function gaussian(x::Real,σ::Real,L::Int64)
    #Input: position,width, L. The centers are uniformly distributed across 0,L.
    #Output: vector responses of L neurons, such that the sum of squared responses is normalized to 1
    u = exp.(-(x.-(1:L)).^2/(2*σ^2)); u ./= sqrt(sum(u.^2))
    return u
end

function vonMises(x::Real,σ::Real,L::Int64)
     u=exp.((1.0/(2*π*σ/L)^2)*(cos.((x.-(1:L))*2π/L).-1)); u ./= sqrt(sum(u.^2))
     return u
 end
 function vonMises(x::Real,σ::Real,L::Int64)
      u=exp.((1.0/(2*π*σ./L)^2)*(cos.((x.-(1:L))*2π/L).-1)); u ./= sqrt(sum(u.^2))
      return u
  end

function gaussian(x::Real,σ::Array,L::Int64)
    #Input: position,width, L. The centers are uniformly distributed across 0,L.
    #Output: vector responses of L neurons, such that the sum of squared responses is normalized to 1
    u = exp.(-(x.-(1:L)).^2 ./(2*σ.^2)); u ./= sqrt(sum(u.^2))
    return u
end

function Network(N::Int64,L::Int64,σ;f1 = gaussian,f2=identity,ϕ=0)
    #Return a network instance with gaussian weights
    W = randn(N,L)
    return Network(W,σ,f1,f2,1.,ϕ)
end
# v = f(W*u(x))/Z
(n::Network)(x) = n.f2.(n.W*(n.f1(x,n.σ,size(n.W)[2] )) .-n.ϕ)./n.Z

function renormalize!(n::Network,R)
    #Renormalization of tuning curves, such that the stdv around 0 is constant
     N,L = size(n.W); x_true = range(1,L,length=500)
    n2 = deepcopy(n);v_true = hcat(n2.(x_true)...); Z = mean(std(v_true,dims=2))
    n.Z= size(n.W)[1]*Z/R
end

function (n::Network)(ntrial::Integer,η;ηu = 0,pn=0)
    #Matrix of ntrial for each of stimuli from 1 to L
     N,L = size(n.W);v = hcat(n.(1:L)...);
     r = zeros(N,ntrial,L);
     if pn==0
     #Uniform Noise
         for x= 1:L
             r[:,:,x] = v[:,x] .+ sqrt.(η).*randn(N,ntrial)
             if ηu !=0
                 r[:,:,x] .+= sqrt(ηu)*n.W*randn(L,ntrial)./n.Z
             end
         end
     elseif pn==1
         #Poisson-like noise in gaussian form
         for x= 1:L
             r[:,:,x] = v[:,x] .+ sqrt.(abs.(v[:,x])).*randn(N,ntrial)
             if ηu !=0
                 r[:,:,x] .+= sqrt(ηu)*n.W*randn(L,ntrial)./n.Z
             end
         end
     end

     return v,r
 end

#Decoding functions


#Ideal Decoder
function ideal_decoder(r,v,η::Float64;iΣ=0,pn=0,circular=0)
    N,L = size(v)
    if (iΣ==0) & (pn==0)
        logl = vec(-0.5*sum((r.-v).^2,dims=1)/η);
    elseif (pn==1) & (iΣ==0)
        #Use poisson likelihood
        logl = vec(-0.5*sum((r.-v).^2 ./abs.(v),dims=1));
    else
        #Use full covariance matrix
        logl = vec(-sum((r.-v).*(iΣ*(r.-v)),dims=1)*0.5)
    end
    likelihood = exp.(logl); likelihood ./=sum(likelihood)
    k_ML = findmax(logl)[2]; k_MSE = sum((1:L).*likelihood)
    if circular==1
        S = sum(sin.(2π*(1:L)/L).*likelihood); C = sum(cos.(2π*(1:L)/L).*likelihood)
        k_MSE = mod(atan(S,C)*L/(2π),L)
    end
    return k_MSE
end


#Poisson ideal decoder
function ideal_decoder_PN(r,v)
    N,L = size(v)
    logl = r.*log(v)
    likelihood = exp.(logl); likelihood ./=sum(likelihood)
    k_ML = findmax(logl)[2]; k_MSE = sum((1:L).*likelihood)
    return k_MSE
end


#Performance evaluation
function evaluate_performance_ideal(n::Network,ntrial::Integer,η::Float64;ηu=0,pn=0,M=0,circular=0)
    #Evaluate mean square error for a given network at a given level of noise, computing
    #RMSE
    N,L = size(n.W);
    v,r = n(ntrial,η,ηu=ηu,pn=pn);  errors = zeros(ntrial,L)
    if ηu!=0
        Σ = η*Matrix(I,N,N) + (ηu*n.W*n.W')/(n.Z^2);iΣ = inv(Σ)
    else
        iΣ=0
    end
    if n.f1==vonMises
        circular=1
    end
    for x=1:L
        for t=1:ntrial
            errors[t,x] = x - ideal_decoder(r[:,t,x],v,η,iΣ=iΣ,pn=pn,circular=circular)
        end
    end
    if circular ==1
        errors = abs.(errors)
        errors[errors.>L/2] .-=L
    end
    RMSE= sqrt(mean(errors.^2))
    return  RMSE #,maximum(abs.(errors))
end

function evaluate_performance_netMMSE(n::Network,ntrial::Integer,η::Float64;M=0,pn=0,ηu=0)
    #MMSE network decoder where ̂x = Σ x p(r|x)/Σ p(r|x)
    N,L = size(n.W);
    v,r = n(ntrial,η);  λ = v'
    if M !=0
        m = L/M
        λ = (hcat(n.(1:m:L)...))';
    else
        m=1
    end
    x = vcat([i*ones(ntrial) for i=1:L]...)
    h = exp.(λ*reshape(r,N,:)/(η)); Z = sum(h,dims=1)
    x_ext = vec((1:m:L)'*h./Z)
    errors = reshape(x-x_ext,ntrial,L);RMSE= sqrt(mean(errors.^2))
    return RMSE
end
