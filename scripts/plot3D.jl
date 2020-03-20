using DrWatson
quickactivate(@__DIR__,"Random_coding")
using Plots, Distributions,JLD
 pyplot(size=(400,300))
 Plots.scalefontsizes(1.1)
 L=500
data = load(datadir("sims/LalaAbbott","Nsigma3D_linear_Nmax=150_Nmin=10_dec=net_η=2_#1.jld"))
RMSE,σVec,NVec = data["RMSE"],data["σVec"],data["NVec"];RMSE .+=1E-9
data2 = load(datadir("sims/LalaAbbott","Nsigma3D_linear_Nmax=150_Nmin=10_dec=net_η=2.jld"))
RMSE_l = data2["RMSE_l"];RMSE_l = mean(RMSE_l,dims=2)[:,:]
RMSE_m = mean(RMSE,dims=3)[:,:]; RMSE_var = var(RMSE,dims=3)[:,:];
CList = reshape( colormap("Blues",length(NVec)),1,: );
xt = ([3,5,7,10,15,20],["3","5","7","10","15","20"])
yt = ([2,3,5,10,25,50],["2","3","5","10","25","50"])
p1=plot(σVec[1:2:end],RMSE_m',linewidth=2,yticks=yt,yaxis=:log,legend=:none,c=CList,grid=:off,xlabel="σ",ylabel="ε")
RMSE_opt,σoptpos_id = findmin(RMSE,dims=2)
σ_opt = [σVec[pos[2]] for pos=σoptpos_id];σ_opt_var = var(σ_opt,dims=3)[:];
σ_opt = mean(σ_opt,dims=3)[:];
RMSE_opt_var = var(RMSE_opt,dims=3)[:]; RMSE_opt = mean(RMSE_opt,dims=3)[:]
p2 = plot(NVec,RMSE_opt,ribbon=sqrt.(RMSE_opt_var),yticks=yt,yaxis=:log,linewidth=2,xlabel = "N",ylabel="ε(σ*)",grid=:off,legend=:none,c=:green)
p3 = plot(NVec,σ_opt,ribbon=sqrt.(σ_opt_var),yaxis=:log,yticks=xt,linewidth=2,xlabel = "N",ylabel="σ*",grid=:off,legend=:none,c=:green)
#p4 = plot(σVec[1:2:end],(RMSE_m./RMSE_l)',linewidth=2;legend=:none,grid=:off,c=CList,xlabel= "σ",ylabel= "ε/ε_l")
data = load(datadir("sims/LalaAbbott","Nsigma3D_linear_Nmax=120_Nmin=10_η=2.jld"))
p4 = plot(NVec,σVec[1:2:end],(RMSE_m./RMSE_l)',xlabel= "N",ylabel="σ",fill=true,colorbar_title="ε/εl")

σtranspos_id = findmin(abs.((RMSE_m./RMSE_l)' .-1),dims=1)[2]
σ_tr = [σVec[1:2:end][pos[1]] for pos=σtranspos_id];
σ_tr[8:end] .= 3.
plot!(NVec,σ_tr',linewidth=1,c=:black,legend=:none)
p=plot(p1,p2,p3,p4)
η = 2.; f="linear";σd = "delta"
name = savename("Nvssigma3D" , (@dict  η f σd),"pdf")
safesave(plotsdir(name) ,p)

#Optimal width
data = load(datadir("sims/LalaAbbott","Nvseta3D_linear_Nmax=210_Nmin=10_σj=10.jld"))
RMSE,ηVec,NVec = data["RMSE"],data["ηVec"],data["NVec"];RMSE .+=1E-9
RMSE_l = data["RMSE_l"]
ηVec = 0.5:0.5:5
RMSE_l = mean(RMSE_l,dims=3)[:,:]; RMSE_var = var(RMSE_l,dims=3)[:,:];
RMSE_m = mean(RMSE,dims=3)[:,:]; RMSE_var = var(RMSE,dims=3)[:,:];
CList = reshape( colormap("Blues",length(NVec)),1,: );
p1=plot(ηVec, RMSE_m',legend=:none,c=CList,grid=:off,xlabel="η")
CList2 = reverse(reshape( colormap("Reds",length(ηVec)),1,: ),dims=2);
yt = ([2,3,5,10,25,50],["2","3","5","10","25","50"])

#data2 = load(datadir("sims/LalaAbbott","Nvseta3D_linear_Nmax=210_Nmin=10_σj=22.jld"))
#RMSE,ηVec,NVec = data2["RMSE"],data["ηVec"],data["NVec"];RMSE .+=1E-9
#RMSE_l2 = data["RMSE_l"]
#RMSE_l = data2["RMSE_l"];RMSE .+=1E-9
#RMSE_l = mean(RMSE_l,dims=3)[:,:]; RMSE_m2 = mean(RMSE2,dims=3)[:,:];
p2= plot(NVec,RMSE_m,legend=:none,yaxis=:log,c=CList2,grid=:off,xlabel="N",ylabel="ε",yticks=yt,linewidth=2)
p3 = plot(1 ./ηVec,NVec,(RMSE_m./RMSE_l),fill=true,xlabel="SNR",ylabel="N",xlim=(0.1,0.5))
f="linear";σj = 22
pyplot(size=(600,300))
name = savename("Nvseta3D" , (@dict  f σj),"pdf")
p=plot(p2,p3)
safesave(plotsdir(name) ,p)
#data3 =  load(datadir("sims/LalaAbbott","Nvseta3D_linear_Nmax=150_Nmin=10_σj=22_#2.jld"))
