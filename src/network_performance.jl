using DrWatson
quickactivate(@__DIR__,"Random_coding")
include(srcdir("network.jl"))
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
        R = hcat([V + sqrt(η)*randn(N,length(x_test)) for t=1:ntrial]...)
        x_t = repeat(x_test,ntrial);

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

#Analytical prediction for error scaling
local_error(N::Int64,σ::Float64,η::Float64) = 2*σ^2*η/N
global_error(N::Int64,σ::Float64,η::Float64;A=1) = A*(1/σ^2)*((1+σ+σ^2)/6)*erfc(sqrt((sqrt(pi)*σ/(sqrt(pi)*σ-2π*σ^2))N/(4(1+η))))
