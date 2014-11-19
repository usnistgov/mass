

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

function make_filters(pulse_model::Vector{Float64}, noise_autocorr::Vector{Float64},
                      npre::Int, f_3db::Float64)
    filters = Dict{String, Vector{Float64}}()
    variance= Dict{String, Float64}()
    
    # filters will be length 5 less than raw data: 1 for allowing shift due to
    # "adjusted triggering", and 4 more for allowing 5-lag parabola fit
    const n = length(pulse_model)-(1+4)
    signal = pulse_model[4:end-2]
    pulse_deriv = signal - pulse_model[3:end-3]

    # 2- and 3-parameter pulse models
    M2 = hcat(signal, ones(Float64, n))
    M3 = hcat(signal, pulse_deriv, ones(Float64, n))

    # True R and downweighted gappy Rd
    R = toeplitz(noise_autocorr[1:n])

    weights = ones(n)
    new_weights = [.01, .01, .2, .4, .6, .8]
    for i = 1:length(new_weights)
        weights[npre+i-1] = new_weights[i]
    end

    # Now the bandpass smoother
    Smooth = Array(Float64, n, n)
    dt = 12.8e-6
    tau = 1/(2*pi*f_3db*dt)
    for i = 1:n
        for j = 1:n
            Smooth[i,j] = exp(-.5*((i-j)/tau)^2)
        end
        Smooth[i,:] = Smooth[i,:] ./ sum(Smooth[i,:])
    end
    

    for use_deriv in [true,false]
        key4 = use_deriv ? "d" : "-"
        model = use_deriv ? M3 : M2

        for use_gap in [true,false]
            key3 = use_gap ? "g" : "-"
            
            key = "--$(key3)$(key4)-"
            gmodel = copy(model)
            if use_gap
                for c in 1:size(gmodel)[2]
                    gmodel[:,c] .*= weights
                end
            end
            RinvM = (R \ gmodel)
            A = model' * RinvM
            this_f = vec((A \ (RinvM'))[1,:])
            if use_gap
                this_f = this_f .* weights
            end
            filters[key] = this_f
            variance[key] = inv(A)[1,1]

            keyf = "f-$(key3)$(key4)-"
            this_f = Smooth * vec((A \ (RinvM)')[1,:])
            if use_gap
                this_f = this_f .* weights
            end
            filters[keyf] = this_f
            #variance[keyf] = inv(A)[1,1]

            keys = "s-$(key3)$(key4)-"
            Smodel = Smooth*gmodel
            RinvM = R \ Smodel
            A = Smodel' * RinvM
            this_f = vec((A \ (RinvM'*Smooth))[1,:])
            if use_gap
                this_f = this_f .* weights
            end
            filters[keys] = this_f
            variance[keys] = inv(A)[1,1]
        end
    end
    println(variance)
    filters
end


function filter_data(filename::String, filters::Dict{String,Vector{Float64}})

    file = MicrocalFiles.LJHFile(filename)
    const np = file.nrec
    nrecs = 2000
    _times = Array(Uint64, nrecs)
    newdata = Array(Uint16, file.nsamp, nrecs)
    nsamp_read = 0

    ph = Dict{String, Array{Float64, 1}}()
    for k in keys(filters)
        klag0 = k[1:1]*"0"*k[3:end]
        kshift = k[1:4]*"t"
        kslag0 = klag0[1:4]*"t"
        ph[k] = Array(Float64, np)
        ph[klag0] = Array(Float64, np)
        ph[kshift] = Array(Float64, np)
        ph[kslag0] = Array(Float64, np)
    end

    SG_filter = [[ -6 24  34 24 -6],
                 [-14 -7   0  7 14],
                 [ 10 -5 -10 -5 10]]./70.0

    nchunks = div(np+nrecs-1, nrecs)
    for i=1:nchunks
        if nrecs + nsamp_read > np
            nrecs = np-nsamp_read
        end
        thesesamples = nsamp_read+1 : nsamp_read+nrecs
        
        MicrocalFiles.fileRecords(file, nrecs, _times, newdata)
        for k in keys(filters)
            filt = (filters[k])[:]'
            klag0 = k[1:1]*"0"*k[3:end]

            output = vcat(filt*newdata[1:end-5,1:nrecs],
                          filt*newdata[2:end-4,1:nrecs],
                          filt*newdata[3:end-3,1:nrecs],
                          filt*newdata[4:end-2,1:nrecs],
                          filt*newdata[5:end-1,1:nrecs],
                          filt*newdata[6:end,  1:nrecs])  # shape is (6,nrecs)
            
            ph[klag0][thesesamples] = output[4,:]
            parab_fits = SG_filter * output[2:6,:]  # rows = [const,slope,quadratic]
            parab_peak = parab_fits[1,:] - parab_fits[2,:].^2 ./ (4*parab_fits[3,:])
            ph[k][thesesamples] = parab_peak

            # Now the shift-by-1 data (shift data later by looking at earlier part of it)
            kshift = k[1:4]*"t"
            kslag0 = klag0[1:4]*"t"
            ph[kslag0][thesesamples] = output[3,:]
            parab_fits = SG_filter * output[1:5,:]  # rows = [const,slope,quadratic]
            parab_peak = parab_fits[1,:] - parab_fits[2,:].^2 ./ (4*parab_fits[3,:])
            ph[kshift][thesesamples] = parab_peak
        end
        
        nsamp_read += nrecs
        println("Done with  $(nsamp_read) samples")
    end
    ph
end


function fit_data(filename::String, filters::Matrix{Float64})
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


function save_32filters(filterdict, ph, pulse_model)
    h5=h5open("/Volumes/Data2014/Data/Data_tupac/20130906_RIKENexpts/20130906_T3B_307filters.hdf5","w")
    for k in keys(filterdict)
        ds_update(h5, "filter"*k, filterdict[k])
    end
    for k in keys(ph)
        ds_update(h5, k, ph[k])
    end
    ds_update(h5, "pulse_model", pulse_model[3:end-2])
    dpdt = pulse_model[3:end-2]-pulse_model[2:end-3]
    ds_update(h5, "pulse_deriv", dpdt)
    close(h5)
end
