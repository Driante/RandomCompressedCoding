using DrWatson
quickactivate(@__DIR__,"Random_coding")
using Plots, Distributions,JLD
pyplot(size=(400,300))
Plots.scalefontsizes(1.2)
 #Plot error in funciton of N and σL=500
data = load(datadir("sims/iidnoise","Nsigma_Nmax=100_Nmin=10_SNR=1_c=1_dec=ideal.jld"))
L=500
RMSE,σVec,NVec = data["RMSE"],data["σVec"],data["NVec"];RMSE .+=1E-9
#First layer theoretical bound
fl = sqrt(2)*σVec/sqrt(L)
RMSE_m = mean(RMSE,dims=3)[:,:]; RMSE_v = var(RMSE,dims=3)[:,:];
RMSE_opt,σoptpos_id = findmin(RMSE,dims=2)
σ_opt = [σVec[pos[2]] for pos=σoptpos_id];σ_opt_var = var(σ_opt,dims=3)[:];
σ_opt = mean(σ_opt,dims=3)[:];
RMSE_opt_var = var(RMSE_opt,dims=3)[:]; RMSE_opt = mean(RMSE_opt,dims=3)[:]
CList = reshape( colormap("Blues",length(NVec)),1,: );
yt= ([0.1,1,10,100],["0.1","1","10","100"])
p1 = plot(σVec,RMSE_m'  .+1E-9,yticks=  yt,xlabel ="σ",yaxis=:log,ylabel = "ε",legend=:none,linewidth=2,grid=:off,c=CList)
#plot!(p1,σ_opt,RMSE_opt,linewidth=2,c=:black)
#plot!(p1,σVec,fl,linewidth=3,c=:black)
RMSE_opt_var[(RMSE_opt.-sqrt.(RMSE_opt_var)).<0] .= 0
p2 = plot(NVec,RMSE_opt,ribbon=sqrt.(RMSE_opt_var),yaxis=:log,linewidth=2,yticks=yt,xlabel = "N",ylabel="ε(σ*)",grid=:off,legend=:none,c=:green)
yt2 = ([1,5,10,20,50],["1","5","10","20","50"])
p3 = plot(NVec,σ_opt,ribbon=sqrt.(σ_opt_var),yaxis=:log,linewidth=2,xlabel = "N",yticks=yt2,ylabel="σ*",grid=:off,legend=:none,c=:green)
CList2 = reverse(reshape( colormap("Reds",length(σVec[1:2:end])),1,: ),dims=2)
p4 = plot(NVec,RMSE_m[:,1:2:end],yaxis=:log,xlabel ="N",yticks=yt,ylabel = "ε",linewidth=2,legend=:none,grid=:off,c=CList2)
p5 = contour(σVec,NVec,log10.(RMSE_m),fill=true,xaxis=:log,yaxis=:log)
plot!(p5,σ_opt,NVec,linewidth=2,c=:black,legend=:none,xlabel="σ",ylabel="N")
p= plot(p1,p4,p2,p3)
SNR = "1"; dec = "ideal";ηu=1;pn=1
name = savename("Nvssigma" , (@dict  SNR dec),"pdf")
safesave(plotsdir(name) ,p)


#SNR vs N plots - Width distribution
data = load(datadir("sims/sigmadist","Nsigmadist_Nmax=200_Nmin=8_dist=Pareto0.01.jld"))
RMSE,SNRVec,NVec = data["RMSE"],data["SNRVec"],data["NVec"];RMSE .+=1E-9
RMSE = mean(RMSE,dims=3)[:,:]; RMSE_v = var(RMSE,dims=3)[:,:]
CList = reshape( colormap("Blues",length(SNRVec)),1,: );
p_RMSE_contour = heatmap(SNRVec,NVec,RMSE,xaxis=:log,yaxis=:log,xticks=10.0.^(-1:1),xlabel = "SNR",ylabel="N",grid=:off)
p_RMSE_lines = plot(NVec,RMSE,yaxis=:log,legend=:none,grid=:off,xlabel="N",ylabel ="ϵ(σ*)",c=CList,linewidth=2)
p=plot(p_RMSE_contour,p_RMSE_lines)
dist="Uniform"
name = savename("NvsSNR" , (@dict dist ),"pdf")
#safesave(plotsdir(name) ,p)
data = load(datadir("sims/sigmadist","Nsigmadist_Nmax=200_Nmin=8_dist=10.jld"))
RMSE2,SNRVec2,NVec2 = data["RMSE"],data["SNRVec"],data["NVec"];RMSE .+=1E-9
RMSE2 = mean(RMSE2,dims=3)[:,:]; RMSE_v2 = var(RMSE2,dims=3)[:,:]
εratio = RMSE./RMSE2
p2 = heatmap(SNRVec,NVec,εratio,clim=(0,2),xlabel="SNR",ylabel="N")
dist2 = "10"
name = savename("sigmadistcomp" , (@dict dist dist2), "pdf")
safesave(plotsdir(name) ,p)


#input noise difference
data = load(datadir("sims/inputnoise","incorrcomp_Nmax=80_Nmin=10_SNR=2_dec=ideal.jld"))
RMSEo,RMSEoi,NVec,σo,σoi = data["RMSEo"],data["RMSEoi"],data["NVec"],data["σo"],data["σoi"]
ηuVec = 0.2/L:0.4/L:2/L
εratio = (RMSEoi-RMSEo)./RMSEo; σratio = (σoi-σo)./σo
p1=heatmap(ηuVec*500,NVec,εratio,xlabel="ηu",ylabel = "N",colorbar_title= "εratio")
p2 = heatmap(ηuVec*500,NVec,σratio,xlabel="ηu",ylabel = "N",colorbar_title="σratio")
p=plot(p1,p2,layout=(2,1))
SNR=2
name = savename("corrvsind ", (@dict SNR ),"pdf")
safesave(plotsdir(name) ,p)

#SNRvsN
data = load(datadir("sims/iidnoise","NvsSNR_Nmax=150_Nmin=10_SNRmax=5_SNRmin=0.3_dec=ideal.jld"))
RMSE_opt,SNRVec,σ,NVec = data["RMSE_opt"],data["SNRVec"],data["σ"],data["NVec"]
SNRVec = [0.3,0.5,0.8,1.,1.3,1.5,1.7,2.,3.,5.]
CList = reshape( colormap("Blues",length(SNRVec)),1,: )
xt = ([0.3,0.5,1.0,2.0,3,5],["0.3","0.5","1.0","2.0","3.0","5.0"])
yt = ([10,20,50,80,150],["10","20","50","80","150"])
p_RMSE_contour = plot(SNRVec,NVec,log10.(RMSE_opt.+1E-7),xaxis=:log10,yaxis=:log10,xticks=xt,xlabel = "SNR",yticks=yt,ylabel="N",grid=:off,fill=true)
#p_RMSE_lines = plot(NVec,RMSE_opt_SNR.+1E-7,yaxis=:log,legend=:none,grid=:off,xlabel="N",ylabel ="ϵ(σ*)",c=CList)
xt2 = ([1,10,20,50],["1","10","20","50"])
p_σ_contour = plot(SNRVec,NVec,log10.(σ),xaxis=:log10,yaxis=:log10,xticks=xt,xlabel = "SNR",yticks=yt,ylabel="N",grid=:off,fill=true)
#p_σ_lines = plot(NVec,σ_opt_SNR,yaxis=:log,legend=:none,grid=:off,xlabel="N",ylabel ="σ*",c=CList)
plot(p_RMSE_contour,p_σ_contour)
dec="ideal"
name = savename("NvsSNR_RMSE ", (@dict dec ),"pdf")
safesave(plotsdir(name) ,p_RMSE_contour)
name = savename("NvsSNR_sigma ", (@dict dec ),"pdf")
safesave(plotsdir(name) ,p_σ_contour)



#Mcomparisons
data = load(datadir("sims","Mcomparison_N=80_SNR=3_dec=MLP.jld"))
MVec,RMSE_id,RMSE,σVec = data["MVec"],data["RMSE_id"],data["RMSE"],data["σVec"]
RMSE_m = mean(RMSE,dims=3)[:,:]; RMSE_v = var(RMSE,dims=3)[:,:]
CList = reshape( colormap("Blues",length(MVec)),1,: );
plot(σVec,RMSE_m',legend=:none,c=CList,linewidth=2,yaxis=:log,grid=:off)
plot!(σVec[2:end],mean(RMSE_id,dims=2)[2:end],linewidth=2,c=:black)
RMSE_opt,σoptpos_id = findmin(RMSE,dims=2)
σ_opt = [σVec[pos[2]] for pos=σoptpos_id];σ_opt_var = var(σ_opt,dims=3)[:];
σ_opt = mean(σ_opt,dims=3)[:];
RMSE_opt_var = var(RMSE_opt,dims=3)[:]; RMSE_opt = mean(RMSE_opt,dims=3)[:]
plot(MVec,σ_opt)
RMSE_opt,Moptpos_id = findmin(RMSE,dims=1)
M_opt = [MVec[pos[1]] for pos=Moptpos_id];M_opt_var = var(M_opt,dims=3)[:];
σ_opt = mean(σ_opt,dims=3)[:];
RMSE_opt_var = var(RMSE_opt,dims=3)[:]; RMSE_opt = mean(RMSE_opt,dims=3)[:]
