using DrWatson
quickactivate(@__DIR__,"Random_coding")
using JLD
include(srcdir("network.jl"))

##Computation of probability of error (and errors) for uncorrelated stimuli

function error_probability(L,N,η;nnets = 8,ntrial=1000)
    #Probability of error for L stimul encoded by N neurons with random responses.
    #The inference is correct if the response is closer to the original point, wrong otherwise.
    pg = zeros(nnets)
    for net=1 : nnets
        #Average over different `network` realizations
        V = randn(N,L);
        errors = zeros(ntrial,L);
        Threads.@threads for t=1:ntrial
            R = V .+sqrt(η)*randn(N,L);
            for x=1:L
                #ML decoder
                x_ext = findmin(vec(sum((R[:,x].-V).^2,dims=1)))[2]
                if x_ext != x
                    pg[net] +=1
                end
                errors[t,x] += abs((x_ext-x)/L)
            end
        end
    end
    pg ./= (ntrial*L)
    @info "L=",L, "N=",N, "η = ",η, "P=", pg
    return pg
end


## Run script with different parameters
#Probability of error vs N  for different N, noise fixed
NVecL = 10:5:70;
LVec = [10,50,100,500,1000];
η = 0.5;
pε_vsL = [hcat(error_probability.(L,NVecL,η)...) for L = LVec];

#Probability of error for different noise magntidues, L fixed
L=500
NVecη = 10:10:140;
ηVec = [0.1, 0.5, 1., 1.5, 2.];
pε_vsη = [hcat(error_probability.(L,NVecη,η)...) for η = ηVec];


name = savename("uncorrelated_stims" , (@dict L η),"jld")
data = Dict("NVecL"=>NVecL ,"NVecη" => NVecη, "LVec" => LVec, "ηVec" => ηVec,"pε_vsL " => pε_vsL  ,"pε_vsη" => pε_vsη)
safesave(datadir("sims/iidnoise/idealdec",name) ,data)
