using DrWatson
quickactivate(@__DIR__,"Random_coding")
using JLD
include(srcdir("network3D.jl"))
function build_grid(x_min, x_max,L)
        Δi = range(x_min,x_max, length=L)
        cVec  = [ [x , y , z]  for x=Δi , y= Δi , z= Δi ]
        return hcat(cVec...)'
end

function compute_tuning_curves_net(W::AbstractArray,σVec,x_test;s=0.1)
        ntest,~ = size(x_test)
        Vdict = Dict();
        for σ = σVec
                n = Network3D(W,σ,x_min,x_max);V = compute_tuning_curves(n,x_test)
                Vdict[σ] = V
                println("Computed tuning curves at σ = $σ")
        end
        name = savename("tuning_curves3D" , (@dict ntest  s ),"jld")
        data = Dict("σVec"=>σVec,"Vdict"=>Vdict ,"x_test" => x_test)
        safesave(datadir("sims/LalaAbbott/tuning_curves",name) ,data)
end
#Script to build tuning curves in the 3D model. To follow LalazarAbbott, receptive field centers tile a cube of 100x100x100.
#Parameters of the network
N=412; L=100;x_min  = -100.; x_max = 100.; M=L^3;
#Build grid of test points
x_test_min = -40; x_test_max=40; ntest = 15; s=0.1;
#Build a grid of test stimuli  with arbitrary precision
x_test = build_grid(x_test_min,x_test_max,ntest);
#Or use stimuli from data
#~,~,tP= import_and_clean_data(); x_test = tP[1]
σVec = 5.:1.:36.;W = sqrt(1/(s*M))*sprandn(N,M,s);
compute_tuning_curves_net(W,σVec,x_test)
