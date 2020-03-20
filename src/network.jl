using DrWatson
quickactivate(@__DIR__,"Random_coding")
#Strutcture and function for 1D model and ML-MSE inference
using Distributions,StatsBase , LinearAlgebra, MultivariateStats,Random,SparseArrays,SpecialFunctions
##Todo::rewrite code properly
#constant of discretization of tuning curves and stimulus bounds
ntest = 500; x_min,x_max=0,1;
test_point(x_min,x_max) = range(x_min,x_max,length = ntest)
Δx = x_max-x_min;x_test=test_point(x_min,x_max)
#Network structures
mutable struct Network
    #1st layer: number of neurons, vector of centers,tuning function and width
    L::Int64; A;σ;cVec;f1::Function;
    #Connectivity matrix, number of neurons of second layer,function, normalization constant
    W::Array{Float64,2};N::Int64; f2::Function; Z::Array{Float64};
    #row by row normalization flag, if ==1, tin computing the tuning curves Z is updated
    rxrnorm
end
function Network(N::Int64,L::Int64,σ::Float64; f1= gaussian ,f2 =identity,rxrnorm = 1)
    #Defaul normalization is maintaining the average over the variance constant
    if f1==gaussian
        A = sqrt(1/(sqrt(π)*σ - 2*π*σ^2));
    elseif f1==VonMises
        A =sqrt(1/( besseli(0,2*1/(2π*σ)^2) - (besseli(0,1/(2π*σ)^2))^2));
    end
    cVec = [Δx*i/L for i=1:L]
    W = sqrt(1/L)*randn(N,L); Z = ones(N)
    return Network(L,A,σ,cVec,f1,W,N,f2,Z,rxrnorm)
end
function Network(N::Int64,Lpop::Array{Int64},σpop::Array{Float64};f1= gaussian ,f2 =identity,rxrnorm =1,c=1)
    L = sum(Lpop);Δx = x_max-x_min;cVec = [[Δx*i/L for i=1:L] for L=Lpop]
    function constraints(σp,Lp,c)
        L = sum(Lp)
        A1 = sqrt(L/(sum(Lp.*(sqrt.(π*σp.^2) .- 2π*σp.^2).*[1,c^2 ])))
        A2= c*A1
        return [A1,A2]
    end
    Apop = constraints(σpop,Lpop,c)
    W = sqrt(1/L)*randn(N,L); Z = ones(N)
    return Network(L,Apop,σpop,cVec,f1,W,N,f2,Z,rxrnorm)
end


function gaussian(x,A,σ,cVec)
    u = A.*exp.(-(x.-cVec).^2 ./(2*σ.^2))
    return u
end
function VonMises(x,A,σ,cVec)
    u = A.*exp.(cos.(2*π*(x.-cVec))/((2*π*σ).^2))
end
function gaussian(x,pop)
    u = A*exp.(-(x.-cVec).^2 ./(2*σ.^2))
    return u
end
function compute_tuning_curves(n::Network,x_test)
    if length(n.σ)==1
        U = n.f1(x_test',n.A,n.σ,n.cVec)
    else
        #case of multiple populations
        U = vcat([n.f1(x_test',A,σ,cVec) for (A,σ,cVec) in zip(n.A,n.σ,n.cVec)]...)
    end
    V = n.f2.(n.W*U)./n.Z
    #row by row normalization
    if n.rxrnorm==1
        n.Z .= vec(sqrt.(var(V,dims=2)))
        V = V./n.Z
    end
    return U,V
end




#Error computing functions
#Ideal decoders
function MMSE_gon(r,V,η::Float64,x_test)
    logl = vec(-0.5*sum((r.-V).^2,dims=1)/η);
    likelihood = exp.(logl); likelihood ./=sum(likelihood)
    x_ext = x_test'*likelihood
    return x_ext
end

function MSE_ideal_gon(n::Network,η::Float64;ntrial=50,MC=0,tol=1E-7)
    #Mean square error with gaussian output noise of fixed variance. If MC ==1
    N=n.N;x_test = test_point(x_min,x_max);U,V = compute_tuning_curves(n,x_test);
    if MC==0
        #Generate ntrial noisy responses and average to obtain the MSE
        ε = 0;R = V .+ sqrt(η)*randn(N,ntest,ntrial)
        for t = 1:ntrial,i=1:ntest
            x = x_test[i]; r = R[:,i,t];
            x_ext = MMSE_gon(r,V,η,x_test);
            ε += (x-x_ext)^2
        end
        ε /= (ntrial*ntest)
    else
        #Montecarlo extimate of the mse
        ε = []; t= 1;s=0
        while t <5000
            R = V .+ sqrt(η)*randn(N,ntest)
            for i=1:ntest
                x = x_test[i]; r = R[:,i];
                x_ext = MMSE_gon(r,V,η,x_test);
                s += (x-x_ext)^2/ntest
            end
            push!(ε,s/(t))
            if t>100
                if std(ε[t-50:t]) < tol ; break;end
            end
            t +=1
        end
    end
    return ε
end


function MSE_net_gon(n::Network,η::Float64;ntrial=50,MC=0,tol=1E-7)
    #Mean square error with gaussian output noise of fixed variance. If MC ==1
    N=n.N;x_test = test_point(x_min,x_max);U,V = compute_tuning_curves(n,x_test);
    #vector of bias
    b = sum(V.^2,dims=1)'
    if MC==0
        #Generate ntrial noisy responses and average to obtain the MSE
        R = hcat([V + sqrt(η)*randn(N,length(x_test)) for t=1:ntrial]...);x_t = repeat(x_test,ntrial);
        H = exp.((V'*R .-0.5*b )/(η));Zh = sum(H,dims=1);H = H./Zh
        x_ext = H'*x_test;
        ε = mean((x_t-x_ext).^2)
    else
        #Montecarlo extimate of the mse
        ε = []; t= 1;s=0
        while t <5000
            R = V .+ sqrt(η)*randn(N,ntest);
            H = exp.((V'*R .-0.5*b )/(η));Zh = sum(H,dims=1);H = H./Zh
            x_ext = H'*x_test;
            s  += mean((x_ext-x_test).^2)
            push!(ε,s/(t))
            if t>100
                if std(ε[t-50:t]) < tol ; break;end
            end
            t +=1
        end
    end
    return ε
end



function errors_table(n::Network,η::Float64,ntrial::Int64)
    #Given a network, a level of noise and a number of trials, return a table of errors. Can be used to plot histograms of errors
    N=n.N
    x_test = test_point(x_min,x_max)
    U,V = compute_tuning_curves(n,x_test)
    R = V .+ sqrt(η)*randn(N,ntest,ntrial)
    errors = zeros(ntrial,ntest)
    for t = 1:ntrial,i=1:ntest
        x = x_test[i]; r = R[:,i,t];
        x_ext = MMSE_gon(r,V,η,x_test);
        errors[t,i]= abs(x-x_ext)
    end
    return errors
end

#Circular variable MMSE
function MMSE_gon_c(r,V,η::Float64,x_test)
    logl = vec(-0.5*sum((r.-V).^2,dims=1)/η); likelihood = exp.(logl);
    S = sum(sin.(2π*x_test')*likelihood); C = sum(cos.(2π*x_test')*likelihood)
    x_ext = mod(atan(S,C)/(2π),1)
    return x_ext
end
function MSE_ideal_gon_c(n::Network,η::Float64;ntrial=50,MC=0,tol=1E-6)
    #Mean square error with gaussian output noise of fixed variance. If MC ==1
    N = n.N;x_test = test_point(x_min,x_max);U,V = compute_tuning_curves(n,x_test);
    if MC==0
        #Generate ntrial noisy responses and average to obtain the MSE
        ε = 0;R = V .+ sqrt(η)*randn(N,ntest,ntrial)
        for t = 1:ntrial,i=1:ntest
            x = x_test[i]; r = R[:,i,t];
            x_ext = MMSE_gon_c(r,V,η,x_test);
            ε += (1 .- cos.(2π*(x-x_ext)))/(2π^2)
        end
        ε /= (ntrial*ntest)
    else
        #Montecarlo extimate of the mse
        ε = []; t= 1;s=0
        while t <10000
            R = V .+ sqrt(η)*randn(N,ntest)
            for i=1:ntest
                x = x_test[i]; r = R[:,i];
                x_ext = MMSE_gon_c(r,V,η,x_test);
                s +=  ((1 .- cos.(2π*(x-x_ext)))/(2π^2))/ntest
            end
            push!(ε,s/(t))
            if t>100
                if abs(ε[t] - ε[t-50]) < tol ; break;end
            end
            println(s/(t))
            t +=1
        end
    end
    return ε
end
function errors_table_c(n::Network,η::Float64,ntrial::Int64)
    #Given a network, a level of noise and a number of trials, return a table of errors. Can be used to plot histograms of errors
    N=n.N;x_test = test_point(x_min,x_max);U,V = compute_tuning_curves(n,x_test)
    R = V .+ sqrt(η)*randn(N,ntest,ntrial)
    errors = zeros(ntrial,ntest)
    for t = 1:ntrial,i=1:ntest
        x = x_test[i]; r = R[:,i,t];
        x_ext = MMSE_gon_c(r,V,η,x_test);
        errors[t,i]=  abs(x-x_ext)
    end
    errors[errors.>1/2] .-=1
    return abs.(errors)
end

function MSE_net_gon_c(n::Network,η::Float64;ntrial=50,MC=0,tol=1E-7)
    #Mean square error with gaussian output noise of fixed variance. If MC ==1
    N=n.N;x_test = test_point(x_min,x_max);U,V = compute_tuning_curves(n,x_test);
    #vector of bias
    b = sum(V.^2,dims=1)'
    if MC==0
        #Generate ntrial noisy responses and average to obtain the MSE
        R = hcat([V + sqrt(η)*randn(N,length(x_test)) for t=1:ntrial]...);x_t = repeat(x_test,ntrial);
        H = exp.((V'*R .-0.5*b )/(η));Zh = sum(H,dims=1);H = H./Zh
        S = sum(H'*sin.(2π*x_test)); C = sum(H'*cos.(2π*x_test'))
        x_ext = mod(atan(S,C)/(2π),1)
        ε = mean((1 .- cos.(2π*(x_t-x_ext)))/(2π^2))
    else
        #Montecarlo extimate of the mse
        ε = []; t= 1;s=0
        while t <5000
            R = V .+ sqrt(η)*randn(N,ntest);
            H = exp.((V'*R .-0.5*b )/(η));Zh = sum(H,dims=1);H = H./Zh
            S = sum(H'*sin.(2π*x_test)); C = sum(H'*cos.(2π*x_test'))
            x_ext = mod(atan(S,C)/(2π),1)
            s  += mean((1 .- cos.(2π*(x_test-x_ext)))/(2π^2))
            push!(ε,s/(t))
            if t>100
                if std(ε[t-50:t]) < tol ; break;end
            end
            t +=1
        end
    end
    return ε
end

#Analytical prediction for error scaling
local_error(N::Int64,σ::Float64,η::Float64) = 2*σ^2*η/N
global_error(N::Int64,σ::Float64,η::Float64;A=1) = A*(1/σ)*((1+σ+σ^2)/6)*erfc(sqrt((sqrt(pi)*σ/(sqrt(pi)*σ-2π*σ^2))N/(4(1+η))))
