using DrWatson
quickactivate(@__DIR__,"Random_coding")
#Strutcture and function for 1D model and ML-MSE inference
#using TensorBoardLogger,Logging
using Plots,LaTeXStrings,JLD2,Statistics
include(srcdir("plot_utils.jl"))
##
mydir= datadir("sims/iidnoise/MLPdec")
η = 0.1
flnames = filter(x ->occursin("decoders_relu_Ml=600_Ms=100_η=$(η)",x),
readdir(mydir))
Nsims = length(flnames)
@info Nsims
ε_id,ε_lin,ε_rf,ε_1l,ε_2l,σVec = [[] for i=1:6]
for f in flnames
    data = load(datadir("sims/iidnoise/MLPdec",f))
    push!(σVec,data["σVec"])
    @info length(data["σVec"])
    dvsσ = data["dvsσ"]
    push!(ε_id,[dvsσ[i][:ε_id] for i = 1:length(σVec[end])])
    push!(ε_lin,[dvsσ[i][:ε_lin] for i = 1:length(σVec[end])])
    push!(ε_rf,[dvsσ[i][:ε_rf] for i = 1:length(σVec[end])])
    push!(ε_1l,[dvsσ[i][:ε_1l] for i = 1:length(σVec[end])])
    push!(ε_2l,[dvsσ[i][:ε_2l] for i = 1:length(σVec[end])])
end

##
ε_id_m = mean(ε_id)
ε_id_s =  make_asymmetric_ribbon(ε_id_m,std(ε_id))
ε_lin_m = mean(ε_lin)
ε_lin_s =  make_asymmetric_ribbon(ε_lin_m,std(ε_lin))
ε_1l_m = mean(ε_1l)
ε_1l_s =  make_asymmetric_ribbon(ε_1l_m,std(ε_1l))
ε_2l_m = mean(ε_2l)
ε_2l_s =  make_asymmetric_ribbon(ε_2l_m,std(ε_2l))
ε_rf_m = mean(ε_rf)
ε_rf_s =  make_asymmetric_ribbon(ε_rf_m,std(ε_rf))

σVec = mean(σVec)

##
p1 = plot(σVec,ε_id_m,ribbon=ε_id_s,m=:o,linewidth=2,yaxis=:log10,
    label="ideal",grid=:off,xlabel=L"$\sigma$",ylabel=L"$\varepsilon^2$",
    fillalpha=0.3,ylim=(1E-5,1E-2),legend=:none)
#plot!(σVec,ε_lin_m,m=:o,label="lin")
plot!(p1,σVec,ε_rf_m,ribbon=ε_rf_s,label="RF",m=:o,
fillalpha=0.3,linewidth=3,)
plot!(σVec,ε_1l_m,ribbon=ε_1l_s,label="MLP1L",m=:o,
fillalpha=0.3,linewidth=3,yaxis=:log10)

plot!(p1,σVec,ε_2l_m,ribbon=ε_2l_s,label="MLP2L",m=:o,
fillalpha=0.3,linewidth=3,)
##
    name = savename("dec_comparison_relu", (@dict  η ),"svg")
safesave(plotsdir("decoder",name) ,p1)