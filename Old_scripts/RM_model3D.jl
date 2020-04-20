using DrWatson
quickactivate(@__DIR__,"Random_coding")
using Distributions,LinearAlgebra, MultivariateStats,Random,SparseArrays
#3D network definition of model v_i = f1(∑ w_ij u_j(x))
mutable struct Network3D
    M :: Int;N::Int #second layer number of neurons
    W #Connectivity Matrix
    σ #Width of 1st layer tuning curve
    PP1l  #Preferred Positions of first layer
    B::Array{Float64,1} #Biases
    Z::Float64#normalization Constant
    f1::Function
    f2::Function
    p:: Int
end

#Create  a grid of preferred positions of first layer cells
function build_PP(x_min, x_max,L)
        Δi = range(x_min,x_max, length=L)
        PP  = [ [x , y , z]  for x=Δi , y= Δi , z= Δi ]
        return PP
end

#u_j(x) ∝ exp(- (x-s_j)^2/(2*σ^2))
function linear_gaussian(x,σ::Float64,PP;p=2)
        #Activity of neurons with center in centers and width σwhen a stimulus x is presented
        Δs2 = map((j) ->sum((j-x).^2),vec(reshape(PP,1,:)))
        u = exp.(-Δs2 /(2*(σ^2)))
        u ./= sqrt(sum(u.^2))
        return u
end
function linear_gaussian(x,σ::Array,PP ; p=2)
        #Activity of neurons with center in centers and width σwhen a stimulus x is presented
        Δs2 = map((j) ->sum((j-x).^2),vec(reshape(PP,1,:)))

        if p==2
           u = sqrt(8)*exp.(-Δs2 ./(2*(σ.^2)))./(π*σ.^2).^(3/4)
       else
           u = 8*exp.(-Δs2 ./(2*(σ.^2)))./(2*π*σ.^2).^(3/2)
        end
        return u
end



function Network3D(M::Int,N::Int,σ,x_min,x_max;Z=1,f1 = linear_gaussian,f2 = identity,sparsity=1,threeshold=0,p=2)
    #Create network from #neurons of first layer(must be a third power of L), #neurons of second layer, width
    L = Int64(round(M^(1/3)))
    #W = randn(N,M)
    if sparsity < 1
        W = sprandn(N,M,sparsity)
        Z = sqrt(sparsity)
    end
    PP1l = build_PP(x_min,x_max,L)
    B = threeshold*ones(N)
    return Network3D(M,N,W,σ,PP1l,B,Z,f1,f2,p)
end


function (net::Network3D)(x)
    u = net.f1(x,net.σ ,net.PP1l,p=net.p)
    v = net.f2.(net.W*u./net.Z .+ net.B)
    return v
end
(net::Network3D)(x_vec ::Array{Float64,2}) = Array(mapslices( x -> net(x) , x_vec ,dims= 2)')

function ideal_decoder3D(rn,r,η,tP)
    logl = vec(-0.5*sum((rn.-r).^2,dims=1)/η);
    likelihood = exp.(logl); likelihood ./=sum(likelihood)
    x_MSE = sum(tP.*likelihood,dims=1)
    return x_MSE
end

function evaluate_performance_ideal3D(r,η,tP,ntrial; n_m=0)
    #generate ntrial noisy response for each point in the stimulus space
    N,L = size(r)
    errors = zeros(ntrial,L)
    for t=1:ntrial
        rn = r .+ sqrt(η)*randn(N,L)
        for p=1:L
            errors[t,p] = sum((tP[p,:] - vec(ideal_decoder3D(rn[:,p],r,η,tP))).^2)
        end
    end
    RMSE = sqrt(mean(errors))
    return RMSE
end

function evaluate_performance_netMMSE(r,η,tP,ntrial)
    N,L = size(r);
    se=0
    for t = 1:ntrial
        rn = r .+ sqrt(η)*randn(N,L)
        for p=1:L
            l = exp.((rn[:,p]'*r)/(η)); l ./= sum(l)
            x_ext = (l*tP)';
            if sum(isnan.(x_ext)).==0
                se += sum((x_ext .- tP[p,:]).^2)
            else
                x_ext = tP[findmax(l)[2][2]]
                se += sum((x_ext .- tP[p,:]).^2)
            end
        end
    end
    RMSE = sqrt(se/(ntrial*L))
    return RMSE
end

function evaluate_performance_linear(r,η,tP,PP,ntrial)
    N,L = size(r);
    se=0
    a = mean((PP*tP'./(r.+1E-9)));se=0
    for t = 1:ntrial
        rn = r .+ sqrt(η)*randn(N,L)
        se += sum((tP -(a*pinv(PP)*rn)').^2)
    end
    RMSE = sqrt(se/(ntrial*L))
    return RMSE
end
