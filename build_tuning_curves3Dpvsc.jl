using DrWatson
quickactivate(@__DIR__,"Random_coding")
using JLD,NNlib
#Compute tuning curves for pure and conjuntive coding of first layer
include(srcdir("network3D.jl"))
include(srcdir("lalazarabbott.jl"))
function build_grid(x_min, x_max,L)
        Δi = range(x_min,x_max, length=L)
        cVec  = [ [x , y , z]  for x=Δi , y= Δi , z= Δi ]
        return hcat(cVec...)'
end

function compute_tuning_curves_pvsc(Wp::AbstractArray,Wc::AbstractArray,σVec,x_test)
        ntest,~ = size(x_test); N,~=size(W)
        Vpdict = Dict();  Vcdict = Dict()
        for σi = σVec
                nc = Network3D(Wc,σi,x_min,x_max,rxrnorm=1);
                @time V = compute_tuning_curves(nc,x_test); Vcdict[σi] = V;
                np = Network3D_p(Wp,σi,x_min,x_max,rxrnorm=1);
                V = compute_tuning_curves(np,x_test); Vpdict[σi] = V
                println("Computed tuning curves at σ = $σi")
        end
        return Vcdict,Vpdict
end

N=400; L=100;x_min  = -1.; x_max = 2.; M=L^3;s=0.1;
x_test_min = -1.; x_test_max=2.; ntest = 20; x_test = build_grid(x_test_min,x_test_max,ntest);
#Or use stimuli from data
#~,~,tP= import_and_clean_data(); x_test = tP[1]; ntest,~ =size(x_test)
σVec  = (1:14)./L; Wc = sqrt(1/(s*M))*sprandn(N,M,s);Wp = sqrt(1/(3*L))*randn(N,3*L);
Vcdict,Vpdict = compute_tuning_curves_pvsc(W,σVec,x_test)
name = savename("tuning_curves3D_pvsc" , (@dict ntest  ),"jld")
data = Dict("σVec"=>σVec,"Vpdict"=>Vpdict ,"Vcdict"=> Vcdict,"x_test" => x_test)
