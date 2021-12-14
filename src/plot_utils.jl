using DrWatson
quickactivate(@__DIR__,"Random_coding")
#Utils for quality figures
using Plots,LaTeXStrings,JLD2,Statistics
pyplot(size=(400,300))
f = Plots.font("sans-serif",12)
f2 = Plots.font("sans-serif",16)
fs = Dict(:guidefont => f2, :legendfont => f, :tickfont => f)
default(; fs...)

function make_asymmetric_ribbon(m,s)
    #Handle ribbon when plotting data in log scale, scuh that 
    #we don't have negative values
    stop = s
    sbottom = s
    sbottom[(m-s).<0] .= m[(m-s).<0] .- 1E-8
    return (sbottom,stop)
end