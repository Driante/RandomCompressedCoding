using DrWatson
quickactivate(@__DIR__,"Random_coding")
using JLD
include(srcdir("network3D.jl"))
include(srcdir("lalazarabbott.jl"))

##Precompute tuning  curves in a Random FFN, in case of conjunctive and pure populations in the first layer
function build_grid(x_min, x_max,L)
        #Build  a grid of test points in a cube x_min,x_max, with L points per side (L^3 points in total)
        Δi = range(x_min,x_max, length=L)
        cVec  = [ [x , y , z]  for x=Δi , y= Δi , z= Δi ]
        return hcat(cVec...)'
end

function compute_tuning_curves_pvsc(Wp::AbstractArray,Wc::AbstractArray,σVec,x_test)
        ntest,~ = size(x_test);
        Vpdict = Dict();  Vcdict = Dict()
        Threads.@threads for σi = σVec
                println("Computing tuning curves at σ = $(σi), on thread ",Threads.threadid())
                nc = Network3D(Wc,σi,x_min,x_max,rxrnorm=1);
                @time V = compute_tuning_curves(nc,x_test); Vcdict[σi] = V;
                np = Network3D_p(Wp,σi,x_min,x_max,rxrnorm=1);
                V = compute_tuning_curves(np,x_test);
                Vpdict[σi] = V
        end
        return Vcdict,Vpdict
end

##Run script with different parameters
N=400; L=1000;x_min  = -0.1; x_max = 1.1; M=L^3;s=0.1;
M =15^3
x_test_min = 0.; x_test_max=1.; ntest = 20;
x_test = build_grid(x_test_min,x_test_max,ntest);
σVec  = (3:15)./L;
#Different
#Wc = sqrt(1/(s*M))*sprandn(N,M,s);Wp = sqrt(1/(3*L))*randn(N,3*L);
Wc = sqrt(1/(M))*randn(N,M);Wp = sqrt(1/(M))*randn(N,M);
Vcdict,Vpdict = compute_tuning_curves_pvsc(Wp,Wc,σVec,x_test)
name = savename("tuning_curves3D_pvsc_Meq" , (@dict ntest  ),"jld")
data = Dict("σVec"=>σVec,"Vpdict"=>Vpdict ,"Vcdict"=> Vcdict,"x_test" => x_test)
safesave(datadir("sims/LalaAbbott/tuning_curves",name) ,data)
