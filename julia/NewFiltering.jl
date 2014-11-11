

using HDF5, PyPlot
using MicrocalFiles, HDF5Helpers, CovarianceModels

#include("NoiseAnalysis.jl")
#include("RandomMatrix.jl")

# Use "/Volumes/Data2014/Data/Data_tupac/20130906_RIKENexpts/20130906_T3B_mass.hdf5"

function analyze_file(hdf5name::String, channum::Int)
    pulse_model, noise_autocorr, npre = load_file(hdf5name, channum)
    #noise_model = make_noise_model(noise_autocorr)
    #filters = make_components(pulse_model, noise_model, npre)
    filters = make_components(pulse_model, noise_autocorr, npre)
    ph,dpdt,baseline,ptmean=filter_data(file.name, filters)

    h5 = h5open(hdf5name, "r+")
    try
        grp = g_create_or_open(h5["chan$(channum)"], "nov10filter")
        ds_update(grp, "pulse_height", ph)
        ds_update(grp, "ph_times_at", dpdt)
        ds_update(grp, "baseline", baseline)
    finally
        close(h5)
    end
end


function load_file(hdf5name::String, channum::Int)
    h5 = h5open(hdf5name, "r")
    try
        npre = read(attrs(h5)["npresamples"])
        grp = h5["chan$(channum)"]
        pulse_model = grp["average_pulse"][:]
        pulse_model[1:npre+1] = 0
        pulse_model = convert(Vector{Float64}, pulse_model)
        pulse_model /= maximum(pulse_model)
        noise_autocorr = grp["noise_autocorr"][:]
    finally
        close(h5)
    end
    return pulse_model, noise_autocorr, npre
end


function make_noise_model(noise_autocorr::Vector{Float64})
    amps,bases = fit_exponential_model(noise_autocorr, 5)
    CovarianceModel(amps, bases, length(noise_autocorr))
end


function make_componentsXXXSHIFTED(pulse_model::Vector{Float64},
                                   noise_model::CovarianceModel,
                                   npre::Int)
    n = length(pulse_model)-1
    dpds = pulse_model[2:end] - pulse_model[1:n]
    M=hcat(pulse_model[1:end-1], dpds, ones(Float64, n))

    M[npre+3,:] *= .01
    M[npre+4,:] *= .01
    M[npre+5,:] *= .2
    M[npre+6,:] *= .5

    WM = Array(Float64, size(M))
    for c in 1:size(M)[2]
        WM[:,c] = CovarianceModels.whiten(noise_model, M[:,c])
    end
    
    A = WM' * WM
    @show inv(A)
    # Use the following filter by (1) reweighting the data, (2) whitening it, and (3)
    # taking inner product of that with the following.
    AinvWMT = A \ WM'

    filters = zeros(Float64, size(AinvWMT))
    for i=1:n
        d = zeros(Float64, n)
        if i==npre+3
            d[i] = .01
        elseif i==npre+4
            d[i] = .01
        elseif i==npre+5
            d[i] = .2
        elseif i==npre+6
            d[i] = .5
        else
            d[i] = 1.0
        end
        d = CovarianceModels.whiten(noise_model, d)
        filters[:,i] = AinvWMT*d
    end
    filters
end


function make_components(pulse_model::Vector{Float64}, noise_autocorr::Vector{Float64},
                         npre::Int)
    n = length(pulse_model)-1
    dpds = pulse_model[2:end] - pulse_model[1:n]
    M=hcat(pulse_model[2:end], dpds, ones(Float64, n))

    R = toeplitz(noise_autocorr[1:n])
    R[npre+2,:] *= 100
    R[npre+3,:] *= 100
    R[npre+4,:] *= 5
    R[npre+5,:] *= 2
    R[:,npre+2] *= 100
    R[:,npre+3] *= 100
    R[:,npre+4] *= 5
    R[:,npre+5] *= 2
    RinvM = (R \ M)
    A = M' * RinvM
    @show inv(A)
    # Use the following filter by (1) reweighting the data, (2) whitening it, and (3)
    # taking inner product of that with the following.
    filters = A \ (RinvM)'
    filters
end


function filter_data(filename::String, filters::Matrix{Float64})
    file = MicrocalFiles.LJHFile(filename)
    const np = file.nrec
    nrecs = 1000
    _times = Array(Uint64, nrecs)
    newdata = Array(Uint16, file.nsamp, nrecs)
    nsamp_read = 0

    ph = Array(Float64, np)
    dpdt = Array(Float64, np)
    baseline = Array(Float64, np)
    ptmean = Array(Float64, np)

    nchunks = div(np+nrecs-1, nrecs)
    for i=1:nchunks
        if nrecs + nsamp_read > np
            nrecs = np-nsamp_read
        end
        MicrocalFiles.fileRecords(file, nrecs, _times, newdata)
        filter_result = filters * newdata[2:end,:]
        ph[nsamp_read+1:nsamp_read+nrecs] = filter_result[1,1:nrecs]
        dpdt[nsamp_read+1:nsamp_read+nrecs] = filter_result[2,1:nrecs]
        baseline[nsamp_read+1:nsamp_read+nrecs] = filter_result[3,1:nrecs]
        ptmean[nsamp_read+1:nsamp_read+nrecs] = mean(newdata[1:file.npre,1:nrecs], 1)
        nsamp_read += nrecs

    end
    ph, dpdt, baseline, ptmean
end
