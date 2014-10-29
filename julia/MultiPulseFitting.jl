

using HDF5, PyPlot
using MicrocalFiles, HDF5Helpers, CovarianceModels

include("NoiseAnalysis.jl")
include("RandomMatrix.jl")

# Try on "/Volumes/Data2014/Data/NSLS_data/2012_06_14/2012_06_14_S_chan93.ljh"
# Noise  "/Volumes/Data2014/Data/NSLS_data/2012_06_14/2012_06_14_V_chan93.noi"



# Works only on 1d vectors, not arrays!
function read_complex(ds::HDF5.HDF5Dataset)
    v = read(ds)
    v[:,1] + 1im*v[:,2]
end


# Do all multi-pulse fitting analyses on one data set:
function all_MPF_analysis(setname::String="S", noiseset::String="V", date::String="14")
    const badchan = [3,5,11,13,19,29,33,35,39,53,55,57,61,65,71,87,113]
    const PATH="/Volumes/Data2014/Data/NSLS_data/2012_06_$(date)"
    for cnum=1:2:120
        cnum in badchan && continue
        filename = @sprintf("%s/2012_06_%s_%s_chan%d.ljh", PATH, date, setname, cnum)
        noisename = @sprintf("%s/2012_06_%s_%s_chan%d.noi", PATH, date, noiseset, cnum)
        println("Working on $filename")
        hdf5name = hdf5_name_from_ljh_name(filename, "mpf")
        Noise_analysis(hdf5name, noisename)
    end
    for cnum=1:2:120
        cnum in badchan && continue
        filename = @sprintf("%s/2012_06_%s_%s_chan%d.ljh", PATH, date, setname, cnum)
        println("Working on $filename")
        MPF_analysis(filename)
    end
end

function MPF_analysis_2011(setname::String="C_auto", noiseset::String="C",
                           pulseshape=nothing)
    const PATH="/Volumes/Data2014/Data/Data_CDM/2011_09_12"
    const cnum = 1
    filename = @sprintf("%s/2011_09_12_%s_chan%d.ljh", PATH, setname, cnum)
    noisename = @sprintf("%s/2011_09_12_%s_chan%d.noi", PATH, noiseset, cnum)
    println("Working on $filename noise")
    hdf5name = hdf5_name_from_ljh_name(filename, "mpf")
    Noise_analysis(hdf5name, noisename)
    println("Working on $filename data")
    if pulseshape != nothing
        h5file = h5file_update(hdf5name)
        g = g_create_or_open(h5file, "chan1")
        ds = ds_update(g, "average_pulse", pulseshape)
        close(h5file)
    end
    MPF_analysis(filename)
end


function oct24_work ()
    #@time all_MPF_analysis("ZB", "V", "14")
    #@time all_MPF_analysis("ZA", "V", "14")
    #@time all_MPF_analysis("Z", "V", "14")
    @time all_MPF_analysis("G", "B", "15")
end

function oct27_work()
    @time MPF_analysis_2011("C_auto", "C")
    copy_noise_analysis(
          "/Volumes/Data2014/Data/Data_CDM/2011_09_12/2011_09_12_C_auto_mpf.hdf5",
          "/Volumes/Data2014/Data/Data_CDM/2011_09_12/2011_09_12_D_auto_mpf.hdf5")
    copy_noise_analysis(
          "/Volumes/Data2014/Data/Data_CDM/2011_09_12/2011_09_12_C_auto_mpf.hdf5",
          "/Volumes/Data2014/Data/Data_CDM/2011_09_12/2011_09_12_E_auto_mpf.hdf5")
    copy_noise_analysis(
          "/Volumes/Data2014/Data/Data_CDM/2011_09_12/2011_09_12_C_auto_mpf.hdf5",
          "/Volumes/Data2014/Data/Data_CDM/2011_09_12/2011_09_12_F_auto_mpf.hdf5")
    copy_noise_analysis(
          "/Volumes/Data2014/Data/Data_CDM/2011_09_12/2011_09_12_C_auto_mpf.hdf5",
          "/Volumes/Data2014/Data/Data_CDM/2011_09_12/2011_09_12_G_auto_mpf.hdf5")
    copy_noise_analysis(
          "/Volumes/Data2014/Data/Data_CDM/2011_09_12/2011_09_12_C_auto_mpf.hdf5",
          "/Volumes/Data2014/Data/Data_CDM/2011_09_12/2011_09_12_H_auto_mpf.hdf5")
    copy_noise_analysis(
          "/Volumes/Data2014/Data/Data_CDM/2011_09_12/2011_09_12_C_auto_mpf.hdf5",
                        "/Volumes/Data2014/Data/Data_CDM/2011_09_12/2011_09_12_I_auto_mpf.hdf5")
    pulse_shape = 0
    h5file = h5file_update("/Volumes/Data2014/Data/Data_CDM/2011_09_12/2011_09_12_C_auto_mpf.hdf5")
    g = g_create_or_open(h5file, "chan1")
    pulse_shape = g["average_pulse"][:]
    close(h5file)
    @time MPF_analysis_2011("D_auto", "C", pulse_shape)
    @time MPF_analysis_2011("E_auto", "C", pulse_shape)
    @time MPF_analysis_2011("F_auto", "C", pulse_shape)
    @time MPF_analysis_2011("G_auto", "C", pulse_shape)
    @time MPF_analysis_2011("H_auto", "C", pulse_shape)
    @time MPF_analysis_2011("I_auto", "C", pulse_shape)
end

# Analyze a noise set for use in MPF analysis later

function Noise_analysis(hdf5name::String, noisename::String, forceNew::Bool=false)
    println("We are about to compute MPF results and store into '$hdf5name'")
    h5file = h5file_update(hdf5name)
    try
        channum=split(split(noisename,"_chan")[end], ".")[1]
        grpname=string("chan$channum")
        h5grp = g_create_or_open(h5file, grpname)

        if forceNew || ! exists(h5grp, "noise")
            println("\nComputing noise analysis on chan $(channum)...")
            println(noisename)
            data = MicrocalFiles.fileData(noisename)
            nlags, nexps = 500, 1 # For 2011 TDM data
            #deblip_nsls_data!(data, -8)
            #nlags, nexps = 88, 3 # For 2012 NSLS data
            summary = compute_noise_summary(data, nlags, nexps)
            plot_noise_summary(summary)
            title("Channel $(channum) noise model")
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
            println("Bases are ",summary.exp_bases)
            println("Amps  are ",summary.exp_amplitudes)
        end
    finally
        close(h5file)
    end
end


# Copy all channels' noise analysis results from oldhdf5name to
# newhdf5name (both are HDF5 files.)

function copy_noise_analysis(oldhdf5name::String, newhdf5name::String)
    oldh5 = h5open(oldhdf5name, "r")
    h5file = h5file_update(newhdf5name)
    try
        for channum in 1:2:120
            cname = string("chan$(channum)")
            if !(cname in names(oldh5))
                continue
            end
            grpname=string("chan$(channum)/noise")
            oldsumm = oldh5[grpname]
            summgrp = g_create_or_open(h5file,"chan$(channum)/noise")

            for key in names(attrs(oldsumm))
                attr_update(summgrp, key, a_read(oldsumm, key))
            end
            for dskey in names(oldsumm)
                ds_update(summgrp, dskey, read(oldsumm[dskey]) )
            end
        end
    finally
        close(h5file)
        close(oldh5)
    end
end



# Try to do the entire Multi-Pulse Fitting analysis starting with a pulse file
# and a raw file. Assumes Noise_analysis(noisename) is already done

function MPF_analysis(filename::String, forceNew::Bool=false)
    file = MicrocalFiles.LJHFile(filename)
    hdf5name = hdf5_name_from_ljh_name(filename, "mpf")
    println("We are about to compute MPF results; store into\n$hdf5name")
    h5file = h5file_update(hdf5name)
    try
        const do_all_trigtimes = false
        const do_avg_pulse = false
        const do_all_fits = true

        grpname=string("chan$(file.channum)")
        h5grp = g_create_or_open(h5file, grpname)

        # Find trigger times
        if forceNew || ! exists(h5grp, "trigger_times") | do_all_trigtimes
            trig_times = find_all_pulses(file)
            ds_update(h5grp, "trigger_times", trig_times)
        end
        trigger_times = read(h5grp["trigger_times"])

        # Analyze the average pulse
        if forceNew || ! exists(h5grp, "average_pulse") | do_avg_pulse
            println("\nComputing average pulse shape...")
            avg_pulse = compute_average_pulse(file, trigger_times)
            ds_update(h5grp, "average_pulse", avg_pulse)
            #attr_update(h5grp["average_pulse"], "n")
        end

        # Do the fits
        if forceNew || ! exists(h5grp, "mpf") | do_all_fits
            avg_pulse = read(h5grp["average_pulse"])
            MAXSAMP = 200000  # 8000 for NSLS
            covar_model = CovarianceModel(read_complex(h5grp["noise/model_amplitudes"]),
                                          read_complex(h5grp["noise/model_bases"]),
                                          MAXSAMP)

            println("Computing multi-pulse fits...\n")
            fitter = MultiPulseFitter(avg_pulse, covar_model)
            extendMPF!(fitter, MAXSAMP)
            ph, dph, resid, baseline = multi_pulse_fit_file(file, trigger_times, fitter)

            mpfgrp = g_create_or_open(h5grp, "mpf")
            ds_update(mpfgrp, "pulse_heights", ph)
            ds_update(mpfgrp, "Dpulse_heights", dph)
            ds_update(mpfgrp, "fit_residuals", resid)
            ds_update(mpfgrp, "baseline", baseline)
        end
    finally
        close(h5file)
    end
    nothing
end


function find_all_pulses(file::MicrocalFiles.LJHFile, threshold=-99.9)
    println("Computing trigger times on file $(file.name)...")
    MicrocalFiles.LJHRewind(file)
    const segsize = 500
    times = zeros(Uint64, segsize)
    d = zeros(Uint16, file.nsamp*segsize)

    filter = [.4, .1, -.2, -.5, -.8, 1]
    history = zeros(Float64, length(filter)-1)

    records_read = 0
    trigger_times = Int64[]
    while records_read < file.nrec
        nrec = records_read+segsize > file.nrec ? file.nrec-records_read : segsize

        MicrocalFiles.fileRecords(file, nrec, times, d)
        Ndata = nrec*file.nsamp
        #deblip_nsls_data!(d, -8)
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
            threshold = 10*mad # NSLS 6*mad
        end

        skip_until = 0
        for i in 2:Ndata
            i<skip_until && continue
            if x[i]>threshold && d[i] > 50+data_med
                push!(trigger_times, i+records_read*file.nsamp)
                skip_until = i+10
            end
        end

        history = d[end-4:end]
        records_read += nrec
        println("Read $records_read of $(file.nrec) data records for trigger analysis")
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
    println("Searching $nrecs records for pulses to average")

    MicrocalFiles.LJHRewind(file)
    _times = Array(Uint64, nrecs)
    data = Array(Uint16, file.nsamp, nrecs)
    MicrocalFiles.fileRecords(file, nrecs, _times, data)
    data = reshape(data, length(data))
    #deblip_nsls_data!(data, -9)

    # Find which trigger times are isolated
    #PRE_PERIOD, PRE_DELAY, POST_DELAY = 100, 400, 600  # NSLS
    PRE_PERIOD, PRE_DELAY, POST_DELAY = 100, 30000, 30000
    isolated_times = Integer[]
    for i in 2:npulses-1
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

    # If you can't find ANY isolated pulses, then you are screwed, but here
    # is a dummy (boxcar) filter you can have instead.
    if length(isolated_times) < 1
        avg_pulse = zeros(Float64, PRE_PERIOD+POST_DELAY)
        avg_pulse[PRE_PERIOD:PRE_PERIOD+50] = 1.0
        return avg_pulse
    end
    
    # From the isolated pulses, do an SVD and throw out the outliers.
    u,w,v = find_svd_randomly(pulses, 10)
    badness = sum(v[:,2:end].^2, 2)
    avg_pulse = zeros(Float64, PRE_PERIOD+POST_DELAY)
    np = 0
    critical_badness = 8*median(badness)
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


type MultiPulseFitter
    pulse_model                 ::Vector{Float64}
    dpdt_model                  ::Vector{Float64}
    white_const                 ::Vector{Float64}
    white_slope                 ::Vector{Float64}
    covar_model                 ::CovarianceModel


    function MultiPulseFitter(pulse_model::Vector, covar_model::CovarianceModel,
                              chunklength::Integer=5120)

        const npm = length(pulse_model)
        pulse_model = pulse_model / maximum(pulse_model) # Normalize model to max=1
        dpdt = similar(pulse_model)
        dpdt[2:end] = pulse_model[2:end]-pulse_model[1:end-1]
        dpdt[2] = dpdt[3]
        dpdt[1] = 0

        fitter = new(pulse_model, dpdt, Array(Float64,0),
                     Array(Float64,0), covar_model)
        extendMPF!(fitter, chunklength)
        fitter
    end
end


# Alternate constructor using an HDF5 data source filename
function MultiPulseFitter(hdf5name::String, channum::Integer)
    file = h5open(hdf5name, "r")
    channame = string("chan",channum)
    noisegrp = file[channame*"/noise"]
    const MAXLEN = 100000 # NSLS was 5120
    covar_model = CovarianceModel(read_complex(noisegrp["model_amplitudes"]),
                                  read_complex(noisegrp["model_bases"]), MAXLEN)
    avgpulse = read(file[channame*"/average_pulse"])
    close(file)
    MultiPulseFitter(avgpulse, covar_model, MAXLEN)
end



# Extend the MultiPulseFitter so it can take longer data chunks
function extendMPF!(mpf::MultiPulseFitter, chunklength::Integer)
    if chunklength <= length(mpf.white_const)
        return
    end
    if chunklength > mpf.covar_model.max_length
        @show chunklength, mpf.covar_model.max_length, length(mpf.white_const)
        error("Cannot extend the MultiPulseFitter longer than the CovarianceModel")
    end

    # Whiten baseline functions to their new length
    mpf.white_const = CovarianceModels.whiten(mpf.covar_model, ones(Float64, chunklength))
    mpf.white_slope = CovarianceModels.whiten(mpf.covar_model, linspace(-1, 1, chunklength))
    return mpf
end


# Perform multi-pulse fitting on a length of data. Note that the pulse arrival
# times vector has to be (1) computed already and (2) be referenced to the
# sample numbers within vector data. Normally that means that the number of
# samples in the file that precede the data vector must be subtracted from
# the pulse_times vector.

function multi_pulse_fit(data::Vector, pulse_times::Vector{Int64},
                         mpf::MultiPulseFitter)

    const Nd = length(data)
    const Np = length(pulse_times)
    const Nonpulse_param = 2
    const Nparam = Np+Nonpulse_param
    @assert maximum(pulse_times) <= Nd
    if Nd > length(mpf.white_const)
        extendMPF!(mpf, Nd)
    end

    # Build the (whitened) components of the data model
    model_components = zeros(Float64, Nd, Nparam)
    model_components[:, 1] = mpf.white_const[1:Nd]
    model_components[:, 2] = mpf.white_slope[1:Nd]
    for i = 1:Np
        pstart = pulse_times[i]-2
        pend = min(pstart+length(mpf.pulse_model)-1, Nd)
        pstart = max(1, pstart)
        model_components[pstart:pend, i+Nonpulse_param] =
            CovarianceModels.whiten(mpf.covar_model, mpf.pulse_model[1:pend-pstart+1])
    end

    # Compute the model's "design matrix".
    # The following BLAS call seemed like it might be faster, but it was no different.
    # A = BLAS.gemm('T', 'N', model_components, model_components)
    A = model_components' * model_components
    covar = inv(A)

    white_data = CovarianceModels.whiten(mpf.covar_model, data)
    mc_dot_data = (model_components' * white_data)
    param = A \ mc_dot_data
    residual = white_data - model_components * param

    return param, covar, norm(residual)
end


# A temporary replacement for multi_pulse_fit which makes a plot of the
# data, model, and residual.

function multi_pulse_fit_with_plot(data::Vector, pulse_times::Vector{Int64},
                                   mpf::MultiPulseFitter)
    param, covar, resid = multi_pulse_fit(data, pulse_times, mpf)
    clf()
    ax2 = subplot(212)
    model = ones(Float64, length(data))*param[1]
    slope = linspace(-1, 1, length(mpf.white_slope))[1:length(data)]*param[2]
    model += slope
    plot(model, color="k")
    sampnum = 1:length(model)
    @show param
    for i=1:length(pulse_times)
        first = pulse_times[i]-2
        last = min(first+length(mpf.pulse_model)-1, length(data))
        np = last+1-first
        component = param[i+2]*copy(mpf.pulse_model[1:np])
        plot(sampnum[first:last], component)
        model[first:last] += component
    end
    ax1 = subplot(211)
    plot(data, "r", model, "b", data-model+param[1]-50, "g")
    param, covar, resid
end



# MPF action on an entire file.

function multi_pulse_fit_file(file::MicrocalFiles.LJHFile, pulse_times::Vector,
                         mpf::MultiPulseFitter)
    nrecs = 1
    extendMPF!(mpf, (nrecs+1)*file.nsamp)

    const MIN_POST_TRIG_SAMPS = 20000
    # const MIN_POST_TRIG_SAMPS = 200 NSLS
    MicrocalFiles.LJHRewind(file)
    _times = Array(Uint64, nrecs)
    newdata = Array(Uint16, file.nsamp*nrecs)
    data_unused = Uint16[]
    t1,t2 = 0,1
    nsamp_read = 0

    np = length(pulse_times)
    pheights = Array(Float64, np)
    pheight_unc = Array(Float64, np)
    baseline = Array(Float64, np)
    resid = Array(Float64, np)
    np_seen = 0

    for i=1:div(file.nrec, nrecs)
        MicrocalFiles.fileRecords(file, nrecs, _times, newdata)
        nsamp_read += length(newdata)
        if length(data_unused) > 0
            data = vcat(data_unused, newdata)
            data_unused = Uint16[]
        else
            data = copy(newdata)
        end
        #deblip_nsls_data!(data, -8)

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
            cut_out_samples = last_samp - (pulse_times[t2-1] - 50)
            last_samp -= cut_out_samples
            data_unused = vcat(data[end-cut_out_samples:end], data_unused)
            data = data[1:end-cut_out_samples]
            t2 -= 1
            t2==t1 && break
        end
        t2 <= t1 && continue  # There were no pulses to fit for

        ncp = t2-t1
        current_times = pulse_times[t1:t2-1] - (last_samp-length(data))
        param, covar, this_resid = multi_pulse_fit(float(data), current_times, mpf)
        #param,covar,this_resid = multi_pulse_fit_with_plot(float(data), current_times, mpf)
        #i>=11 && error("End of test on section $(i)!")
        pheights[np_seen+1:np_seen+ncp] = param[3:2+ncp]
        pheight_unc[np_seen+1:np_seen+ncp] = sqrt(diag(covar)[3:2+ncp])
        resid[np_seen+1:np_seen+ncp] = this_resid
        baseline[np_seen+1:np_seen+ncp] = param[1]
        np_seen += ncp
        if mod(i,1000) == 0
            println("$i chunks seen, with $(np_seen) pulses fit.")
        end
    end
    pheights, pheight_unc, resid, baseline
end



# demo
ignoreme = """
include("MultiPulseFitting.jl");
filename="/Volumes/Data2014/Data/NSLS_data/2012_06_14/2012_06_14_S_chan99.ljh";
data=MicrocalFiles.fileData(filename, 3);
f = h5open("/Volumes/Data2014/Data/NSLS_data/2012_06_14/2012_06_14_S_mpf.hdf5", "r");
tt=f["chan99/trigger_times"][:];
mpf=MultiPulseFitter("/Volumes/Data2014/Data/NSLS_data/2012_06_14/2012_06_14_S_mpf.hdf5", 
     99);
"""
