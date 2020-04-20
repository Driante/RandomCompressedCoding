using DrWatson
quickactivate(@__DIR__,"Random_coding")
using JLD,NNlib
#Pre-compute tuning curves, compute the complexity and R2.
include(srcdir("network3D.jl"))
include(srcdir("lalazarabbott.jl"))
function build_grid(x_min, x_max,L)
        Δi = range(x_min,x_max, length=L)
        cVec  = [ [x , y , z]  for x=Δi , y= Δi , z= Δi ]
        return hcat(cVec...)'
end

function compute_tuning_curves_net(W::AbstractArray,σVec,x_test;s=0.1,B=0,f2=identity)
        ntest,~ = size(x_test); N,~=size(W)
        Vdict = Dict();
        Threads.@threads for σi = σVec
                n = Network3D(W,σi,x_min,x_max,f2=f2,rxrnorm=1); n.B = B*ones(N)
                @time V = compute_tuning_curves(n,x_test)
                Vdict[σi] = V
                println("Computed tuning curves at σ = $σi")
        end
        return Vdict
end
#Script to build tuning curves in the 3D model. To follow LalazarAbbott, receptive field centers tile a cube of 100x100x100.
#Parameters of the network
N=412; L=100;x_min  = -100.; x_max = 100.; M=L^3;s=0.1;
#Build grid of test points
x_test_min = -40.; x_test_max=40.; ntest = 21; x_test = build_grid(x_test_min,x_test_max,ntest);
#Or use stimuli from data
~,~,tP= import_and_clean_data(); x_test = tP[1]; ntest,~ =size(x_test)
σVec  = 5.:36.; W = sqrt(1/(s*M))*sprandn(N,M,s);
#Change non linearity in the second layer
f2=identity; B=0
Vdict = compute_tuning_curves_net(W,σVec,x_test,f2=f2)
#Compute complexity histogram, R2 fit
complexity_sims,R2_sims = [zeros(N,length(σVec)) for n=1:2];
sims_pca = Dict()
for (i,σ) = enumerate(σVec)
    V = Vdict[σ];
    complexity_sims[:,i] = complexity_opt(V,x_test,36);
     ~,~,R2_sims[:,i] =  linear_fit(V,x_test)
    m_pca = fit(PCA,V,pratio=0.999);
    sims_pca[σ] = m_pca
end
f2n = string(f2)
name = savename("tuning_curves3D_01" , (@dict ntest  s f2n B),"jld")
data = Dict("σVec"=>σVec,"Vdict"=>Vdict ,"x_test" => x_test,"complexity_sims"=>complexity_sims,"R2_sims"=>R2_sims,"sims_pca" => sims_pca)
safesave(datadir("sims/LalaAbbott/tuning_curves",name) ,data)
