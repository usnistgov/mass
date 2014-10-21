

using HDF5, PyPlot
using MicrocalFiles, HDF5Helpers

include("NoiseAnalysis.jl")

# Try on "/Volumes/Data2014/Data/NSLS_data/2012_06_14/2012_06_14_S_chan93.ljh"
# Noise  "/Volumes/Data2014/Data/NSLS_data/2012_06_14/2012_06_14_V_chan93.noi"


# type MultiPulseFitter
#
#     noise_autocorr    ::Vector{Float64}
#     nlags             ::Integer
#
#     nexps             ::Integer
#     exp_bases         ::Vector{Complex128}
#     exp_amplitudes    ::Vector{Complex128}
#     model_autocorr    ::Vector{Float64}
#     model_rms         ::Float64
#
#
#     function NoiseSummary(nlags::Integer, nexps::Integer)
#         noise_autocorr = Array(Float64, nlags)
#         model_autocorr = Array(Float64, nlags)
#         exp_bases = Array(Complex128, nexps)
#         exp_amplitudes = Array(Complex128, nexps+1)
#         new(noise_autocorr, nlags, nexps, exp_bases, exp_amplitudes,
#             model_autocorr, 0.0)
#     end
# end


# Works only on 1d vectors, not arrays!
function read_complex(ds::HDF5.HDF5Dataset)
    v = read(ds)
    v[:,1] + 1im*v[:,2]
end


# Do all multi-pulse fitting analyses on one data set:
function all_MPF_analysis(setname::String="S", noisename::String="V")
    const badchan = [3,5,11,13,19,29,33,35,39,55,57,65,71,87,113]
    const nowbad = [73]#, 77, 83]
    const PATH="/Volumes/Data2014/Data/NSLS_data/2012_06_14"
    for cnum=101:2:120
        cnum in badchan && continue
        cnum in nowbad && continue
        filename = @sprintf("%s/2012_06_14_S_chan%d.ljh", PATH, cnum)
        noisename = @sprintf("%s/2012_06_14_V_chan%d.noi", PATH, cnum)
        println("Working on $filename")
        MPF_analysis(filename, noisename)
    end
end

# Try to do the entire Multi-Pulse Fitting analysis starting with a pulse file
# and a raw file.

function MPF_analysis(filename::String, noisename::String, forceNew::Bool=false)
    file = MicrocalFiles.LJHFile(filename)
    hdf5name = hdf5_name_from_ljh_name(filename, "mpf")
    println("We are about to compute MPF results and store into '$hdf5name'")
    h5file = h5file_update(hdf5name)
    try
        grpname=string("chan$(file.channum)")
        h5grp = g_create_or_open(h5file, grpname)

        # Find trigger times
        if forceNew || ! exists(h5grp, "trigger_times")
            trig_times = find_all_pulses(file)
            ds_update(h5grp, "trigger_times", trig_times)
        end
        trigger_times = read(h5grp["trigger_times"])

        # Analyze the average pulse
        if forceNew || ! exists(h5grp, "average_pulse")
            println("\nComputing average pulse shape...")
            avg_pulse = compute_average_pulse(file, trigger_times)
            ds_update(h5grp, "average_pulse", avg_pulse)
            #attr_update(h5grp["average_pulse"], "n")
        end
        avg_pulse = read(h5grp["average_pulse"])

        # Analyze the noise
        if forceNew || ! exists(h5grp, "noise")
            println("\nComputing noise analysis...")
            data = MicrocalFiles.fileData(noisename)
            deblip_nsls_data!(data, -8)
            nlags, nexps = 2048, 2
            summary = compute_noise_summary(data, nlags, nexps)
            summgrp = g_create_or_open(h5grp,"noise")

            # Store basic information
        #     attr_update(h5grp, "npulses", summary.nrec)
        #     attr_update(h5grp, "nsamples", summary.nsamp)
        #     attr_update(h5grp, "npresamples", summary.npre)
        #     attr_update(h5grp, "frametime", summary.dt)
            attr_update(summgrp, "rawname", noisename)
            attr_update(summgrp, "nlags", summary.nlags)
            attr_update(summgrp, "nexps", summary.nexps)
            attr_update(summgrp, "model_rms", summary.model_rms)
            ds_update(summgrp, "autocorrelation", summary.noise_autocorr)
            ds_update(summgrp, "model_autocorr", summary.model_autocorr)
            ds_update(summgrp, "model_amplitudes", summary.exp_amplitudes)
            ds_update(summgrp, "model_bases", summary.exp_bases)
        end
        noise_model = NoiseModel(read_complex(h5grp["noise/model_bases"]),
                                 read_complex(h5grp["noise/model_amplitudes"]),
                                 10240)

        # Do the fits
        if forceNew || ! exists(h5grp, "mpf")
            println("Computing multi-pulse fits...\n")
            fitter = MultiPulseFitter(avg_pulse, noise_model)
            ph, dph, resid = multi_pulse_fit_file(file, trigger_times, fitter)

            mpfgrp = g_create_or_open(h5grp, "mpf")
            ds_update(mpfgrp, "pulse_heights", ph)
            ds_update(mpfgrp, "Dpulse_heights", dph)
            ds_update(mpfgrp, "fit_residuals", resid)
        end


        return trigger_times, noise_model, avg_pulse
    finally
        close(h5file)
    end
end


function find_all_pulses(file::MicrocalFiles.LJHFile, threshold=-99.9)
    println("Computing trigger times on file $(file.name)...")
    MicrocalFiles.LJHRewind(file)
    const segsize = 1000
    times = zeros(Uint64, segsize)
    data = zeros(Uint16, file.nsamp, segsize)

    filter = [.4, .1, -.2, -.5, -.8, 1]
    history = zeros(Float64, length(filter)-1)

    records_read = 0
    trigger_times = Int64[]
    while records_read < file.nrec
        nrec = records_read+segsize > file.nrec ? file.nrec-records_read : segsize

        MicrocalFiles.fileRecords(file, nrec, times, data)
        Ndata = nrec*file.nsamp
        d = reshape(data[:,1:nrec], Ndata)
        if nrec < segsize
            clf()
            plot(d)
        end
        data_med = median(d)

        # Do full convolution, then (1) truncate incomplete trailing values and
        # (2) repair incomplete leading values based on the history array.
        x = conv(d, filter)[1:Ndata]
        for i=1:length(history)
            x[i] = (filter' * [history[i:end], d[1:i]])[1]
        end

        if threshold < 0
            med = median(x[5:end])
            mad = median(abs(x[5:end]-med))
            threshold = 6*mad
        end

        for i in 2:Ndata
            if x[i]>threshold && x[i-1] < threshold &&
                d[i] > 25+data_med
                push!(trigger_times, i+records_read*file.nsamp)
            end
        end

        history = d[end-4:end]
        records_read += nrec
    end
    n = length(trigger_times)
    dt = file.dt * records_read * file.nsamp
    println("Found $n triggers in $dt seconds of data.")
    trigger_times
end



# NSLS data need to have the "blips" removed before doing noise analysis, because the
# blips drive the HSVD to care about the wrong things. I think.

function deblip_nsls_data!(data::Vector, min_diff::Integer)
    min_diff = abs(min_diff)

    N = length(data)
    for i = 2:N-1
        if data[i]+min_diff < data[i-1]
            data[i] = div(data[i-1]+data[i+1], 2)
        end
    end
end



function compute_average_pulse(file::MicrocalFiles.LJHFile,
                            trigger_times::Vector)
    # Choose and load some raw data
    npulses = min(2000, length(trigger_times))
    last_tt = trigger_times[npulses]
    nrecs = 1+div(last_tt, file.nsamp)
    MicrocalFiles.LJHRewind(file)
    _times = Array(Uint64, nrecs)
    data = Array(Uint16, file.nsamp, nrecs)
    MicrocalFiles.fileRecords(file, nrecs, _times, data)
    data = reshape(data, length(data))
    deblip_nsls_data!(data, -9)

    # Find which trigger times are isolated
    PRE_PERIOD, PRE_DELAY, POST_DELAY = 100, 400, 600
    isolated_times = Integer[]
    for i in 2:npulses
        if trigger_times[i]-trigger_times[i-1] > PRE_DELAY &&
            trigger_times[i+1]-trigger_times[i] > POST_DELAY &&
            trigger_times[i] > PRE_PERIOD && trigger_times[i]+POST_DELAY < length(data)
            push!(isolated_times, trigger_times[i])
        end
    end
    pulses = Array(Uint16, PRE_PERIOD+POST_DELAY, length(isolated_times))

    clf()
    for i = 1:length(isolated_times)
        pulses[:, i] = data[isolated_times[i]-PRE_PERIOD+1:isolated_times[i]+POST_DELAY]
        plot(pulses[:,i]+i*5, color=((float(i)/length(isolated_times)),0,0))
    end

    # From the isolated pulses, do an SVD and throw out the outliers.
    u,w,v = find_svd_randomly(pulses, 10)
    badness = sum(v[:,2:end].^2, 2)
    critical_badness = 10*median(badness)
    avg_pulse = zeros(Float64, PRE_PERIOD+POST_DELAY)
    np = 0
    for i=1:length(isolated_times)
        if badness[i] < critical_badness
            avg_pulse += pulses[:, i]
            np += 1
        end
    end

    println("Computed avg pulse from $np clean pulses out of $(length(isolated_times)) isolated.")
    avg_pulse = avg_pulse / float(np)
    avg_pulse -= mean(avg_pulse[1:PRE_PERIOD-3])
    avg_pulse[PRE_PERIOD-2:end]
end



function multi_pulse_fit_version1(data::Vector, pulse_times::Vector, pulse_model::Vector,
                        noise::NoiseModel)
    @assert maximum(pulse_times) < length(data)
    Nd = length(data)
    Np = length(pulse_times)
    mp = whiten(noise, pulse_model)
    model_components = zeros(Float64, Nd, Np+2)
    model_components[:,end-1] = whiten(noise, ones(Float64, Nd))
    model_components[:,end] = whiten(noise, linspace(-1,1,Nd))
    for i = 1:Np
        pstart = pulse_times[i]-2
        pend = min(pstart+length(mp)-1, Nd)
        model_components[pstart:pend, i] = mp[1:pend-pstart+1]
    end

    wdata = whiten(noise, data)
    param = pinv(model_components) * wdata
    println(param)
    wmodel = model_components*param

    clf()
    model = 0.0*wmodel
    model += param[end-1] + param[end] * linspace(-1,1,Nd)
    for i = 1:Np
        pstart = pulse_times[i]-2
        pend = min(pstart+length(mp)-1, Nd)
        model[pstart:pend] += param[i] * pulse_model[1:pend-pstart+1]
    end
    plot(data,"r")
    plot(model,"k")
end


type MultiPulseFitter
    pulse_model                 ::Vector{Float64}
    dpdt_model                  ::Vector{Float64}
    white_pulse                 ::Vector{Float64}
    white_dpdt                  ::Vector{Float64}
    white_const                 ::Vector{Float64}
    white_slope                 ::Vector{Float64}
    noise_model                 ::NoiseModel


    function MultiPulseFitter(pulse_model::Vector, noise_model::NoiseModel)
        np = length(pulse_model)
        nmax = noise_model.max_white_length

        pulse_model = pulse_model / maximum(pulse_model) # Normalize!
        dpdt = similar(pulse_model)
        dpdt[2:end] = pulse_model[2:end]-pulse_model[1:end-1]
        dpdt[2] = dpdt[3]
        dpdt[1] = 0

        # Whiten some functions
        long_pulse_model = [pulse_model, zeros(Float64, nmax-length(pulse_model))]
        long_dpdt = [dpdt, zeros(Float64, nmax-length(pulse_model))]
        wpm = whiten(noise_model, long_pulse_model)
        wdpdt = whiten(noise_model, long_dpdt)
        wc = whiten(noise_model, ones(Float64, nmax))
        wsl = whiten(noise_model, linspace(-1, 1, nmax))

        new(pulse_model, dpdt, wpm, wdpdt, wc, wsl, noise_model)
    end
end


# Perform multi-pulse fitting on a length of data. Note that the pulse arrival
# times vector has to be (1) computed already and (2) be referenced to the
# sample numbers within vector data. Normally that means that the number of
# samples in the file that precede the data vector must be subtracted from
# the pulse_times vector.

function multi_pulse_fit(data::Vector, pulse_times::Vector,
                         mpf::MultiPulseFitter)

    @assert maximum(pulse_times) <= length(data)
    const Nd = length(data)
    const Np = length(pulse_times)
    const Nparam = Np+2
    model_components = zeros(Float64, Nd, Nparam)
    model_components[:,end-1] = mpf.white_slope[1:Nd]
    model_components[:,end] = mpf.white_const[1:Nd]
    for i = 1:Np
        pstart = pulse_times[i]-2
        pend = min(pstart+length(mpf.pulse_model)-1, Nd)
        pstart = max(1, pstart)
        model_components[pstart:pend, i] = mpf.white_pulse[1:pend-pstart+1]
    end

    # Compute the model's state matrix.
    # The following would be correct but slower:
    # A = model_components' * model_components
    A = Array(Float64, Nparam, Nparam)
    first_nonzero_val = ones(Integer, Nparam)
    for i=1:Np
        first_nonzero_val[i] = max(1, pulse_times[i]-2)
    end
    for i=1:Nparam
        for j=1:i
            fnzv = max(first_nonzero_val[i], first_nonzero_val[j])
            A[i,j] = dot(model_components[fnzv:end,i], model_components[fnzv:end,j])
            A[j,i] = A[i,j]
        end
    end

    covar = inv(A)
    w = Array(Float64, Nd)
    @time wdata = whiten(mpf.noise_model, data, w)

    # Correct but slower would be:  mc_dot_data = (model_components' *wdata)
    mc_dot_data = Array(Float64, Nparam)
    for i=1:Nparam
        fnzv = first_nonzero_val[i]
        mc_dot_data[i] = dot(model_components[fnzv:end,i], wdata[fnzv:end])
    end
    param = A \ mc_dot_data

    # Correct but slower would be: residual = wdata - model_components * param
    residual = wdata
    for i=1:Nparam
        fnzv = first_nonzero_val[i]
        residual[fnzv:end] -= model_components[fnzv:end,i] * param[i]
    end
    param, covar, sqrt(mean(residual.^2))
end


# MPF action on an entire file.

function multi_pulse_fit_file(file::MicrocalFiles.LJHFile, pulse_times::Vector,
                         mpf::MultiPulseFitter)
    nrecs = 5
    const MIN_POST_TRIG_SAMPS = 200
    MicrocalFiles.LJHRewind(file)
    _times = Array(Uint64, nrecs)
    datasq = Array(Uint16, file.nsamp, nrecs)
    data_unused = Uint16[]
    t1,t2 = 0,1
    nsamp_read = 0

    np = length(pulse_times)
    pheights = Array(Float64, np)
    pheight_unc = Array(Float64, np)
    resid = Array(Float64, np)
    np_seen = 0

    for i=1:div(file.nrec, nrecs)
        MicrocalFiles.fileRecords(file, nrecs, _times, datasq)
        data = reshape(datasq, length(datasq))
        nsamp_read += length(data)
        if length(data_unused) > 0
            data = vcat(data_unused, data)
            data_unused = Uint16[]
        end

        # Select times from the pulse_times list
        t1 = t2
        while t2 <= np && pulse_times[t2] < nsamp_read
            t2 += 1
        end
        t2 <= t1 && continue  # There were no pulses to fit for

        # Is t2 too close to the end of the vector? If so, back off and save some
        # data for later. Remove one (or more) triggers from the trigger set AND
        # trim down the data vector.
        last_samp = nsamp_read
        while last_samp - pulse_times[t2-1] < MIN_POST_TRIG_SAMPS
            cut_out_samples = last_samp - (pulse_times[t2-1] - 10)
            last_samp -= cut_out_samples
            data_unused = vcat(data[end-cut_out_samples:end], data_unused)
            data = data[1:end-cut_out_samples]
            t2 -= 1
            t2==t1 && break
        end
        t2 <= t1 && continue  # There were no pulses to fit for

        ncp = t2-t1
        current_times = pulse_times[t1:t2-1] - (last_samp-length(data))
        param, covar, this_resid = multi_pulse_fit(data, current_times, mpf)
        pheights[np_seen+1:np_seen+ncp] = param[1:ncp]
        pheight_unc[np_seen+1:np_seen+ncp] = sqrt(diag(covar)[1:ncp])
        resid[np_seen+1:np_seen+ncp] = this_resid
        np_seen += ncp
        if mod(i,100) == 0
            println("$i chunks seen, with $(np_seen) pulses fit.")
        end
    end
    pheights, pheight_unc, resid
end

