#Function useful to analyze the data from Lalazar and Abbott.
using DrWatson
quickactivate(@__DIR__,"Random_coding")
using MAT, Distributions, StatsBase , LinearAlgebra, LsqFit, Random

#Custom nanmean functions to not count Nan elements in data
nanmean(x) = mean(filter(!isnan,x))
nanmean(x,y) = mapslices(nanmean,x,dims=y)
nanmaximum(x) = maximum(filter(!isnan,x))
nanmaximum(x,y) = mapslices(nanmaximum,x,dims=y)
nanvar(x) = var(filter(!isnan,x))
nanvar(x,y) = mapslices(nanvar,x,dims=y)
nansum(x) = sum(filter(!isnan,x))
nansum(x,y) = mapslices(nansum,x,dims=y)

function import_and_clean_data()
        #Import data, clean them, and return noise traces
        data = matread(datadir("exp_raw/LalazarAbbottVaadia_PLOSCB_2016_Dataset.mat"))
        #Read files
        targetPositions = data["targetPositions"]; std = data["singleTrialData"]; EMG = data["meanEMG"]
        #r contains ALL the firing rates  for Nneurons recorder in Ntrials. The target position associated at
        #each trial is described by an index contained in pôsture. This indices range from 1 to 27 and point to a row
        #in targetPostions[nsession], which contains 3D coordinates
        firingRates = std["firingRates"]; posture = std["armPostureIndex"]
        Nsessions = length(firingRates); nPos = size(targetPositions)[1]
        r, var_r ,tPosxN,η,trials = [[] for n=1:5] ;
        for session=1:Nsessions
                #Average firing rate in the same condition (no difference between pronation and supination)
                average = hcat([mean(firingRates[session][: , (posture[session][:,2] .==p )  ],dims=2) for p=1:nPos]...);
                variance = hcat([var(firingRates[session][:,(posture[session][:,2] .==p) ],dims=2) for p=1:nPos]...)
                #Build for every neuron an asosciated vector of positions (mainly, they are always equal except some situations)
                ns = size(firingRates[session])[1]
                ηs= zeros(ns,27,1000); txp = zeros(ns,27)
                for p=1:27
                        rp = firingRates[session][: , (posture[session][:,2] .==p )  ]; ntrial = size(rp)[2];
                        txp[:,p] = ones(ns)*ntrial
                        for t=1:1000
                            for n=1:ns
                                if ntrial > 1
                                    k,l = sample(1:ntrial,2,replace=false)
                                    ηs[n,p,t] = (rp[n,l]-rp[n,k])/sqrt(2)
                                end
                            end
                        end
                end
                push!(η,ηs); push!(trials,txp)
                for n=1:size(average)[1]
                        tPosxN = push!(tPosxN,targetPositions[:,:,session])
                end
                r = push!(r,average); var_r = push!(var_r,variance)
        end
        #Select only neurons with an average of 1 spike per condition (???)
        r = vcat(r...) ; var_r = vcat(var_r...) ; η = cat(η...,dims=1);trials = vcat(trials...)
        #tuned_neurons = findall(vec(nansum(r,2).>30))
        tuned_neurons = findall(vec(nansum(r.>=5,2).>0))
        r = r[tuned_neurons,:] ; var_r = var_r[tuned_neurons,:] ;
        η = η[tuned_neurons,:,:];trials = trials[tuned_neurons,:];
        targetPositions = tPosxN[tuned_neurons]
        return r,var_r, targetPositions,η,trials
end

function linear_fit(r,tP;up=0)
        #Return, from linear fitting parameters, r_linear(s) and PreferedPositions for each neuron (normalized to 1),
        #together with R2 values for the fit. Input: firing rate for each neuron and target positions
        N,nPos = size(r)
        r_linear = [] ; PreferedPositions = zeros(N,3)  ; R2 = zeros(N);baseline= zeros(N)
        #Model + initial guesses
        @. model(x, p) = p[1] + p[2]*x[:,1] + p[3]*x[:,2] + p[4]*x[:,3]; p0 = [1.0,1.0,1.0,1.0]
        for n=1:N
                if sum(isnan.(r[n,:])) ==0
                        rn = r[n,:] ;tpn = tP[n][:,:]
                else
                        rn = r[n,.!isnan.(r[n,:])] ; tpn =  tP[n][.!isnan.(r[n,:]),:]
                end
                myfit = curve_fit(model, tpn, rn, p0)
                PreferedPositions[n,:]= myfit.param[2:end]./sqrt.(sum((myfit.param[2:end]).^2))
                #Projection on Prefered Position of target Positions
                #tPos_norm = tpn*PreferedPositions[n
                #Fit also non registered positions, in such a way that we can fill Nan data with lienar fitiing
                push!(r_linear,model(tP[n],myfit.param))
                #push!(tp_proj,tPos_norm)
                SSres = sum(myfit.resid.^2);SStot = sum((rn.-mean(rn)).^2)
                R2[n] = 1-SSres/SStot
        end
        return r_linear, PreferedPositions, R2
end


nantakeout(r) =  r[findall(vec(sum(isnan.(r),dims=2).==0)),:]
complexity(r::Array{Float64,1},tP,D) = std([abs(r[i] - r[j])/sqrt(sum((p1.-p2).^2)) for (i,p1) = enumerate(eachrow(tP)) , (j,p2) =enumerate( eachrow(tP))  if sqrt(sum((p1.-p2).^2)) <=D && i<j] )
function complexity(r_vec::Array{Float64,2},tP,D)
        #Function that compute complexity as defined in the paper, using as minimum distance D
        N = length(tP)
        cmplx = zeros(N)
        for n=1:N
                cmplx[n] = complexity(r_vec[n,:],tP[n],D)
        end
        return cmplx
end
function complexity_opt(r::Array{Float64,1},nn,d)
    drdx = zeros(length(nn))
    for p = 1:length(nn)
        i,j = Tuple(nn[p])
        if j < i
            drdx[p] = abs(r[i] - r[j])/(d[p])
        end
    end
    return std(drdx[drdx.>0])
end
function complexity_opt(r_vec::Array{Float64,2},tP,D)
    N,np = size(r_vec)
    cmplx = zeros(N)
    d = hcat([sqrt.(sum((tP[1][i,:]'.-tP[1]).^2,dims=2)) for i=1:np]...)
    nn = findall(d.<=D); d =d[nn]
    for n=1:N
        cmplx[n] = complexity_opt(r_vec[n,:],nn,d)
    end
    return cmplx
end
#TODO
#-sample noise traces,add them to the mean responses and decode the position(ML) for linear
#and non linear, see the difference in function of number of neurons
#- optimize paramters (Z and ξ) to match R2 and complexity
