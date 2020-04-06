using DrWatson
quickactivate(@__DIR__,"Random_coding")
using Distributions,LinearAlgebra, MultivariateStats,Random,SparseArrays
#3D network definition of model v_i = f1(∑ w_ij u_j(x))
mutable struct Network3D
    #First layer properties
    M :: Int; A::Float64; σ; cVec; f1::Function
    #Connectivity matrix, number of neurons of second layer,function, normalization constant
    W :: AbstractArray;N::Int64; f2::Function; Z::Array{Float64}; B::Array{Float64}
    #row by row normalization flag, if ==1, tin computing the tuning curves Z is updated
    rxrnorm
end
function Network3D(N::Int64,L::Int64,σ,x_min::Float64,x_max::Float64;s=0.1,f1= gaussian_3D,f2 = identity,rxrnorm=0)
    #Create  a grid of preferred positions of first layer cells
    function build_cVec(x_min, x_max,L)
            Δi = range(x_min,x_max, length=L)
            cVec  = [ [x , y , z]  for x=Δi , y= Δi , z= Δi ]
            return hcat(cVec...)
    end
    V = (x_max-x_min)^3;M=L^3
    A =sqrt(1/((π*σ^2)^(3/2)/V - (2*π*σ^2)^3/V^2))
    cVec = build_cVec(x_min,x_max,L);
    W = sqrt(1/(s*M))*sprandn(N,M,s); Z= ones(N); B=zeros(N)
    return Network3D(M,A,σ ,cVec,f1,W,N,f2,Z,B,rxrnorm)
end

function Network3D(W::AbstractArray,σ ::Float64,x_min::Float64,x_max::Float64;s=0.1,f1= gaussian_3D,f2 = identity,rxrnorm=0)
    N,M = size(W); L = Int(round(M^(1/3)))
    #Create  a grid of preferred positions of first layer cells
    function build_cVec(x_min, x_max,L)
            Δi = range(x_min,x_max, length=L)
            cVec  = [ [x , y , z]  for x=Δi , y= Δi , z= Δi ]
            return hcat(cVec...)
    end
    Vol = (x_max-x_min)^3;M=L^3
    A =sqrt(1/((π*σ^2)^(3/2)/Vol - (2*π*σ^2)^3/Vol^2))
    cVec = build_cVec(x_min,x_max,L);
    Z= ones(N); B=zeros(N)
    return Network3D(M,A,σ ,cVec,f1,W,N,f2,Z,B,rxrnorm)
end

function Network3D_p(W::AbstractArray,σ ::Float64,x_min::Float64,x_max::Float64;f1= gaussian_3D_p,f2 = identity,rxrnorm=0)
    #Network of pure cells
    N,M = size(W); L = Int(round(M/3))
    #Create  a grid of preferred positions of first layer cells
    cVec = range(x_min,x_max, length=L)
    l = (x_max-x_min)
    A =sqrt(1/((π*σ^2)^(1/2)/l - (2*π*σ^2)/l^2))
    Z= ones(N); B=zeros(N)
    return Network3D(M,A,σ ,cVec,f1,W,N,f2,Z,B,rxrnorm)
end

function gaussian_3D_p(x::Array{Float64},A::Float64,σ,cVec)
    u = vcat([A*exp.(-(x[i].-cVec).^2/(2*σ^2)) for i=1:3]...)
    return u
end

function gaussian_3D(x::Array{Float64},A::Float64,σ,cVec::Array{Float64,2})
    d2 = (sum((x.-cVec).^2,dims=1))';u = A*exp.(-d2./(2*(σ.^2)))
    return u
end


function compute_tuning_curves(n::Network3D,x_test)
    #Vector of position is given in a column nposx3
    np,~ = size(x_test);
    V = hcat([n.f2.(n.W*n.f1(x_test[p,:],n.A,n.σ,n.cVec) .+n.B) for p=1:np]...)
    if n.rxrnorm==1
        n.Z = std(V,dims=2); n.Z[n.Z .== 0.] .=1.
        V./=n.Z
    end
    return V
end

function MSE_net_gon(V::Array{Float64},η::Float64,x_test;ntrial=50,MC=0,tol=0.5,maxiter=500)
    N,ntest = size(V)
    #Montecarlo extimate of the mse
    ε = []; t= 1;s=0
    b = sum(V.^2,dims=1)'
    while t <maxiter
        R = V .+ sqrt(η)*randn(N,ntest);
        H = exp.((V'*R .-0.5*b )/(η));Zh = sum(H,dims=1);H = H./Zh
        x_ext = H'*x_test;
        s  += mean(sum((x_ext-x_test).^2,dims=2))
        push!(ε,s/(t))
        if t>30
            if std(ε[t-10:t]) < tol ; break;end
        end
        #println(ε[t])
        t +=1
    end
    return ε
end


function errors_table(V::Array{Float64,2},x_test,η::Float64,ntrial::Int64)
    #Given a network, a level of noise and a number of trials, return a table of errors. Can be used to plot histograms of errors
    N,ntest = size(V);
    R = V .+ sqrt(η)*randn(N,ntest,ntrial)
    errors = zeros(ntrial,ntest)
    b = sum(V.^2,dims=1)'
    for t = 1:ntrial
        R = V .+ sqrt(η)*randn(N,ntest);
        H = exp.((V'*R .-0.5*b )/(η));Zh = sum(H,dims=1);H = H./Zh
        x_ext = H'*x_test;
        errors[t,:]= sqrt.(sum((x_ext-x_test).^2,dims=2))'
    end
    return errors
end

function MSE_net_linear(V,η,x_test,PP;ntrial=50,MC=0,tol=1,maxiter=500)
    #Decoder for a linear tuning curve. Input: noise, principal directions
    N,ntest = size(V);
    #Compute proportionality constant.
    a = mean(x_test*PP[1,:]./(V[1,:].+1E-9));ε = []; t= 1;s=0
    while t<maxiter
        R= V .+ sqrt(η)*randn(N,ntest);
        x_ext = (a*pinv(PP)*R)'
        s  += mean(sum((x_ext-x_test).^2,dims=2))
        push!(ε,s/(t))
        if t>30
            if std(ε[t-20:t]) < tol ; break;end
        end
        t +=1
    end
    return ε
end

function MSE_net_gonD(V::Array{Float64},ηVec::Array{Float64},x_test;ntrial=50,MC=0,tol=0.5,maxiter=500)
    #Decoder where each neuron is affected by gaussian noise with different variance
    N,ntest = size(V)
    #Montecarlo extimate of the mse
    ε = []; t= 1;s=0
    b = sum(V.^2 ./(2*ηVec),dims=1)'
    while t <maxiter
        R = V .+ sqrt.(ηVec).*randn(N,ntest);
        H = exp.((V./ηVec)'*R .-b );Zh = sum(H,dims=1);H = H./Zh
        x_ext = H'*x_test;
        s  += mean(sum((x_ext-x_test).^2,dims=2))
        push!(ε,s/(t))
        if t>30
            if std(ε[t-10:t]) < tol ; break;end
        end
        #println(ε[t])
        t +=1
    end
    return ε
end
