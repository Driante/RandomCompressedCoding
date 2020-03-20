using DrWatson
quickactivate(@__DIR__,"Random_coding")
#Strutcture and function for 1D model and ML-MSE inference
using Distributions,StatsBase , LinearAlgebra, MultivariateStats,Random,SparseArrays
include(srcdir("network.jl"))

function optimal_sigma(n::Network,σVec::Array{Float64},η::Float64,fdec::Function;ntrial=50,MC=0)
    ε = zeros(length(σVec)); x_t = repeat(x_test,ntrial)
    for (i,σi) = enumerate(σVec)
        n.σ = σi;n.A = sqrt(1/(sqrt(π)*σi - 2*π*σi^2))
        ε[i] = sqrt(fdec(n,η,ntrial=ntrial))
    end
    ε_o,i_o= findmin(ε); σ_o = σVec[i_o]
    return ε,ε_o,σ_o
end
N,L = 25,500 ; σi = 10/L; η=0.5
n = Network(N,L,σi,rxrnorm=0);
σVec = collect((1:2:40)./L)
#compute optimal sigma for a single width
ε,ε_o,σ_o = optimal_sigma(n,σVec,η,MSE_ideal_gon,ntrial=200)

#define constraint for the population of first layer
function constraints(σp,Lp,c)
    L = sum(Lp)
    A2 = sqrt(L/(sum(Lp.*(sqrt.(π*σp.^2) .- 2π*σp.^2).*[c^2,1])))
    A1= c*A2
    return [A1,A2]
end

#compute error for different σ1-σ2 pairs and different ratio of contribution A_1/A_2
Lp = [250,250];σp = [5/L,40/L]; cVec = 0.5:0.5:3
ε2 = zeros(length(σVec),length(σVec),length(cVec))
np = Network(N,Lp,σp,rxrnorm=0); np.W = n.W;
for (i,σ1) = enumerate(σVec),j = i:length(σVec)
        σ2 = σVec[j]
        σp = [σ1,σ2];np.σ = σp;
        for (k,c) = enumerate(cVec)
            #compute A according to specific constraints
            Ap = constraints(σp,Lp,c); np.A = Ap
            ε2[i,j,k] = sqrt(MSE_ideal_gon(np,η,ntrial=100))
            println(σ1," ",σ2," ",c)
        end
end

# name = savename("doubleσ" , (@dict Lp ),"jld")
#  data = Dict("ε"=>ε ,"ε2" =>ε2,"σVec"=> σVec,"cVec" =>cVec)
#   safesave(datadir("sims/doublesigma",name) ,data)

p1=plot(σVec,σVec,log10.(ε2[:,:,3]),fill=:true,c=:viridis,xlabel="σ2",ylabel="σ2")
p2 = plot(σVec,[ε,ε2[15,:,2]],yaxis=:log,linewidth=1.5,labels = ["σ_u" "σ1 = 0.06"],xlabel="σ",ylabel = "ε")
p = plot(p1,p2,layout = (2,1))
safesave(plotsdir("tmp","sigma2.png"),p)

#average over network realization
nNet = 8;ε,ε2 = [zeros(length(σVec),nNet) for i=1:2]; c= 0.4
Lp = [400,100];σ1=28/L
for net = 1:nNet
    n = Network(N,L,σi,rxrnorm=0);
    #compute optimal sigma for a single width
    ε[:,net],ε_o,σ_o = optimal_sigma(n,σVec,η,MSE_ideal_gon,ntrial=100)
    np = Network(N,Lp,σp,rxrnorm=0); np.W = n.W;
    for (i,σ2) = enumerate(σVec)
        σp = [σ1,σ2];np.σ = σp;Ap = constraints(σp,Lp,c); np.A = Ap
        ε2[i,net] = sqrt(MSE_ideal_gon(np,η,ntrial=100))
    end
    println(net)
end
ε_m,ε_v = mean(ε,dims=2),var(ε,dims = 2);
ε2_m,ε2_v = mean(ε2,dims=2),var(ε2,dims=2)
