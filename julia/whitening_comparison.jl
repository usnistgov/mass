"""
Multi-pulse fitting done on FAKE data to compare performance of
various whitening models.

Joe Fowler, April 2015

"""

# Key parameters to control the multi-pulse fitting
BG_FIT_SLOPE = false
DO_WITH_PLOTS =  false

using HDF5, PyPlot
cd("/Users/fowlerj/Software/mass/julia")
#using MicrocalFiles, HDF5Helpers, CovarianceModels

#include("NoiseAnalysis.jl")
#include("RandomMatrix.jl")
using fowlerModels



function simulatechunk(pm::PulseModel, nm::ARMAModel, chunklen::Int, expectednp, clearhistory::Bool=false)
    expected_wait_time = float(chunklen)/expectednp
    wts = Int[0]
    while true
        expdev = -expected_wait_time * log(rand())
        wt = int(expdev)
        if wt+wts[end] > chunklen
            break
        end
        push!(wts, wt+wts[end])
    end
    wts=wts[2:end]

    data = generateARMA(nm, chunklen)
    if clearhistory
        pm.prevampl *= 0
    end
    data += generatePulse(pm, chunklen, wts)
    return data, wts
end

function fitchunk(data::Vector, pulseshape::Vector, arrivaltimes::Vector)
    np = length(arrivaltimes)
    N = length(data)
    @assert length(pulseshape) <= N
    M = zeros(Float64, N, 1+np)
    M[1:end,1] = 1.0
    for i=1:np
        nt = N-arrivaltimes[i]+1
        M[arrivaltimes[i]:end, i+1] = pulseshape[1:nt]
    end
    designA = M'*M
    Cpp = inv(designA)
    param = Cpp * (M'*data)
    @show diag(Cpp).^0.5
    @show param
    fit = M * param
    return param,fit
end

function simchunks(pm::PulseModel, nm::ARMAModel, chunklen::Int, expectednp, nchunks::Int=3)
    pm.prevampl *= 0
    pshape = generatePulse(pm, chunklen, [1])
    for i=1:nchunks
        data, arrivaltimes = simulatechunk(pm, nm, chunklen, expectednp, true)
        data += 5000
        param, fit = fitchunk(data, pshape, arrivaltimes)
        @show arrivaltimes
        @show norm(data-fit)
        clf(); plot(data,"b", data-fit,"r")
    end
end

simchunks(pm2, nonoise, 32768, 3, 1)



pm=PulseModel([.983966,.991594,.999480],[-12169.44,-19066.19,31235.63])
pm2=PulseModel([.983966,.991594,.999480,.998601+.00079im, .998601-.00079im],
               [-9776.26,-19066.19,31235.63,-1196.59-4785.09im, -1196.59+4785.09im])
nm = ARMAModel([1.,-0.99], [1,-.5])
nm2 = ARMAModel([1,-2.99335241,2.9867097,-.99335729],sqrt(82.54)*[1,-2.938174902,2.876467758,-.938292854])
nonoise = ARMAModel([1.0],[.01])