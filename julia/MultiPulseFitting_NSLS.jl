"""
Multi-pulse fitting driver.
This is the NSLS-specific version for June 2012 data

Joe Fowler, Autumn 2014

"""

# Key parameters to control the multi-pulse fitting
BG_FIT_SLOPE = false
DO_WITH_PLOTS= false

using HDF5, PyPlot, Dierckx
using MicrocalFiles, HDF5Helpers, CovarianceModels

include("NoiseAnalysis.jl")
include("RandomMatrix.jl")


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
# Try on "/Volumes/Data2014/Data/NSLS_data/2012_06_14/2012_06_14_S_chan93.ljh"
# Noise  "/Volumes/Data2014/Data/NSLS_data/2012_06_14/2012_06_14_V_chan93.noi"


function finite_diff_deriv(input::Vector{Float64})
    dpdt = similar(input)
    dpdt[1] = 0
    dpdt[2:end] = input[2:end] - input[1:end-1]
#    dpdt[1:end-1] = input[2:end] - input[1:end-1]
#    dpdt[end] = dpdt[end-1]
    dpdt
end


function dec3_work ()
    @time all_MPF_analysis("S", "V", "14")
    @time all_MPF_analysis("T", "V", "14")
    @time all_MPF_analysis("X", "V", "14")
    @time all_MPF_analysis("Y", "V", "14")
    @time all_MPF_analysis("Z", "V", "14")
    @time all_MPF_analysis("ZA", "V", "14")
    @time all_MPF_analysis("ZB", "V", "14")
    @time all_MPF_analysis("G", "B", "15")
end


# Analyze a noise set for use in MPF analysis later

function Noise_analysis(hdf5name::String, noisename::String, forceNew::Bool=false)
    h5file = h5file_update(hdf5name)
    try
        channum=split(split(noisename,"_chan")[end], ".")[1]
        grpname=string("chan$channum")
        h5grp = g_create_or_open(h5file, grpname)
        if forceNew || ! exists(h5grp, "noise")
            
            println("\nComputing noise analysis on chan $(channum)...")
            println(noisename)
            data = MicrocalFiles.fileData(noisename)
            deblip_nsls_data!(data, -8)
            nlags, nexps = 88, 3 # For 2012 NSLS data

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


function copy_all_avg_pulses(oldhdf5name::String, newhdf5name::String)
    oldh5 = h5open(oldhdf5name, "r")
    h5file = h5file_update(newhdf5name)
    try
        oldavg = oldh5["chan1/average_pulse"]
        summgrp = g_create_or_open(h5file,"chan1")
        ds_update (summgrp, "average_pulse", read(oldavg))
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
        const do_all_trigtimes = true
        const do_avg_pulse = true
        const do_all_fits = true

        grpname=string("chan$(file.channum)")
        h5grp = g_create_or_open(h5file, grpname)

        # Find trigger times
        if forceNew || ! exists(h5grp, "trigger_times") | do_all_trigtimes
            trig_times = find_all_pulses(file)
            ds_update(h5grp, "trigger_times", trig_times)
            println("The trigger times start with:")
            @show trig_times[1:25]
        end
        trigger_times = read(h5grp["trigger_times"])

        # Analyze the average pulse
        if forceNew || ! exists(h5grp, "average_dpdt") | do_avg_pulse
            println("\nComputing average pulse shape...")
            avg_pulse = compute_average_pulse(file, trigger_times)
            dpdt = finite_diff_deriv(avg_pulse)
            dpdt[1:end-1]=dpdt[2:end]
            ds_update(h5grp, "average_pulse", avg_pulse)
            ds_update(h5grp, "average_dpdt", dpdt)
        end

        # Do the fits
        if forceNew || ! exists(h5grp, "mpf") | do_all_fits
            avg_pulse = read(h5grp["average_pulse"])
            dpdt = read(h5grp["average_dpdt"])
            MAXSAMP = 8000
            covar_model = CovarianceModel(read_complex(h5grp["noise/model_amplitudes"]),
                                          read_complex(h5grp["noise/model_bases"]),
                                          MAXSAMP)
            println("Computing multi-pulse fits...\n")
            
            fitter = MultiPulseFitter(avg_pulse, dpdt, covar_model)
            extendMPF!(fitter, MAXSAMP)

            const DPDT = true
            ph, dpdts, dph, ddpdt, prior_level, residW, residR, baseline =
                multi_pulse_fit_file(file, trigger_times, fitter, DPDT)
            @show dph[1:10]
            
            mpfgrp = g_create_or_open(h5grp, "mpf")
            ds_update(mpfgrp, "pulse_heights", ph)
            ds_update(mpfgrp, "Dpulse_heights", dph)
            if DPDT
                ds_update(mpfgrp, "dpdt", dpdts)
                ds_update(mpfgrp, "Ddpdt", ddpdt)
            end
            ds_update(mpfgrp, "prior_level", prior_level)
            ds_update(mpfgrp, "fit_mahal_dist", residW)
            ds_update(mpfgrp, "fit_residuals", residR)
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
        deblip_nsls_data!(d, -8)
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
            threshold = 5*mad
            clf()
            PyPlot.plt.hist(x[5:end], 1000, histtype="step", color="r")
            println( threshold)
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

        print("Read $records_read of $(file.nrec) data records for trigger analysis.  ")
        println("Found $(length(trigger_times)) so far.")
        # @show trigger_times
        # clf()
        # plot(x, "r")
        # for i =1:length(trigger_times)
        #     plot(trigger_times[i], 20, "ob")
        # end
        # plot(d+100-data_med, "g")
        # error("Oh crap")
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
    deblip_nsls_data!(data, -9)

    # Find which trigger times are isolated
    PRE_PERIOD, PRE_DELAY, POST_DELAY = 100, 400, 600  # NSLS
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
    subplot(111)
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
        if badness[i] > critical_badness
            continue
        end
        avg_pulse += pulses[:, i]
        np += 1
    end
    axes([.6,.6,.3,.3])
    plot(avg_pulse, "k")

    println("Computed avg pulse from $np clean pulses out of $(length(isolated_times)) isolated.")
    avg_pulse = avg_pulse / float(np)
    avg_pulse -= mean(avg_pulse[1:PRE_PERIOD-3])
    avg_pulse = avg_pulse[PRE_PERIOD-2:end]
end


type MultiPulseFitter
    pulse_model                 ::Vector{Float64}
    dpdt_model                  ::Vector{Float64}
    white_const                 ::Vector{Float64}
    white_slope                 ::Vector{Float64}
    covar_model                 ::CovarianceModel


    function MultiPulseFitter(pulse_model::Vector,
                              pulse_deriv::Vector,
                              covar_model::CovarianceModel,
                              chunklength::Integer=5120)

        const npm = length(pulse_model)
        const normalization = 1.0 / maximum(pulse_model) # Normalize model to max=1
        pulse_model = pulse_model * normalization
        dpdt_model  = pulse_deriv * normalization

        fitter = new(pulse_model, dpdt_model, Array(Float64,0),
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
    const MAXLEN = 5120
    covar_model = CovarianceModel(read_complex(noisegrp["model_amplitudes"]),
                                  read_complex(noisegrp["model_bases"]), MAXLEN)
    avgpulse = read(file[channame*"/average_pulse"])
    close(file)
    
    dpdt = finite_diff_deriv(avg_pulse)
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
                         mpf::MultiPulseFitter, use_dpdt::Bool=false)

    const Nd = length(data)
    if maximum(pulse_times) > Nd
        warn("The multi_pulse_fit was called with pulse_times up to $(maximum(pulse_times)) but only $Nd data samples")
        while maximum(pulse_times) > Nd
            pulse_times = pulse_times[1:end-1]
        end
    end
    if Nd > length(mpf.white_const)
        extendMPF!(mpf, Nd)
    end

    const Np = length(pulse_times)
    if BG_FIT_SLOPE
        const Nonpulse_param = 2
    else
        const Nonpulse_param = 1
    end
    if use_dpdt
        nparam_per_pulse = 2
    else
        nparam_per_pulse = 1
    end
    const Nparam = Np*nparam_per_pulse+Nonpulse_param

    # Build the (whitened) components of the data model
    model_components = zeros(Float64, Nd, Nparam)
    if BG_FIT_SLOPE
        model_components[:, end] = mpf.white_slope[1:Nd]
        model_components[:, end-1] = mpf.white_const[1:Nd]
    else
        model_components[:, end] = mpf.white_const[1:Nd]
    end

    # Now one component per pulse (plus optionally a dpdt per pulse)
    for i = 1:Np
        pstart = pulse_times[i]
        pend = min(pstart+length(mpf.pulse_model)-1, Nd)
        pstart = max(1, pstart)

        model_components[pstart:pend, i] =
        CovarianceModels.whiten(mpf.covar_model, mpf.pulse_model[1:pend-pstart+1],
                                pstart-1)[pstart:end]
        if use_dpdt
            model_components[pstart:pend, i+Np] =
            CovarianceModels.whiten(mpf.covar_model, mpf.dpdt_model[1:pend-pstart+1],
                                    pstart-1)[pstart:end]
        end
    end

    # Compute the model's "design matrix".
    A = model_components' * model_components
    covar = inv(A)

    white_data = CovarianceModels.whiten(mpf.covar_model, data)
    mc_dot_data = (model_components' * white_data)
    param = A \ mc_dot_data
    model = model_components * param
    unwhite_model = CovarianceModels.unwhiten(mpf.covar_model, model, length(model))
    residualW = norm(white_data - model)/sqrt(Nd)
    residualR = norm(data - unwhite_model)/sqrt(Nd)

    prior_level = zeros(Np) + param[end-1]
    for i = 1:Np
        t = pulse_times[i]
        if t >= 3 && t <= length(model)
            prior_level[i] = unwhite_model[t-2]
        end
    end
    return param, covar, residualW, residualR, prior_level
end



# A temporary replacement for multi_pulse_fit which makes a plot of the
# data, model, and residual.

function multi_pulse_fit_with_plot(data::Vector, pulse_times::Vector{Int64},
                                   mpf::MultiPulseFitter, use_dpdt::Bool=false)
    param, covar, residW, residR, prior_level =
        multi_pulse_fit(data, pulse_times, mpf, use_dpdt)
    @show pulse_times
    @show param
    @show prior_level
    @show diag(covar).^0.5
    @show length(data), residW, residR
    if BG_FIT_SLOPE
        const Nonpulse_param = 2
    else
        const Nonpulse_param = 1
    end

    model = ones(Float64, length(data))*param[end+1-Nonpulse_param]
    if BG_FIT_SLOPE
        slope = linspace(-1, 1, length(mpf.white_slope))[1:length(data)]*param[end]
        model += slope
    end
    
    clf()
    ax3=subplot(313)
    sampnum = 1:length(model)
    for i=1:length(pulse_times)
        first = pulse_times[i]
        last = min(first+length(mpf.pulse_model)-1, length(data))
        np = last+1-first
        component = param[i]*copy(mpf.pulse_model[1:np])
        if first<1
            shift = 1-first
            first = 1
        else
            shift = 0
        end
        component = component[1+shift:end]
        plot(sampnum[first:last], component)
        model[first:last] += component
        if use_dpdt
            component = param[i+length(pulse_times)]*copy(mpf.dpdt_model[1:np])[1+shift:end]
            plot(sampnum[first:last], component)
            model[first:last] += component
        end
    end
    subplot(311, sharex=ax3)
    plot(data, "r", model, "b"); title("Data and model")
    subplot(312, sharex=ax3)
    plot((data-model), "g"); title("Residuals")
    param, covar, residW, residR, prior_level
end



# MPF action on an entire file.

function multi_pulse_fit_file(file::MicrocalFiles.LJHFile, pulse_times::Vector,
                         mpf::MultiPulseFitter, use_dpdt::Bool=false)
    const nrecs = 1
    extendMPF!(mpf, (nrecs+1)*file.nsamp)
    if pulse_times[1]>10; pulse_times = vcat([2], pulse_times); end
    
    const MIN_PRE_TRIG_SAMPS = 200
    const MIN_POST_TRIG_SAMPS = 200
    if BG_FIT_SLOPE
        const Nonpulse_param = 2
    else
        const Nonpulse_param = 1
    end

    MicrocalFiles.LJHRewind(file)
    _times = Array(Uint64, nrecs)
    newdata = Array(Uint16, file.nsamp*nrecs)
    data_unused = Uint16[]
    t1,t2 = 0,1
    nsamp_read = 0

    np = length(pulse_times)
    pheights = Array(Float64, np)
    dpdts = Array(Float64, np)
    pheight_unc = Array(Float64, np)
    dpdt_unc = Array(Float64, np)
    baseline = Array(Float64, np)
    prior_level = Array(Float64, np)
    residW = Array(Float64, np)
    residR = Array(Float64, np)
    np_seen = 0

    # Store the expected continuation of prior pulses
    tails = zeros(Float64, 600) 

    for i=1:div(file.nrec, nrecs)
        MicrocalFiles.fileRecords(file, nrecs, _times, newdata)
        nsamp_read += length(newdata)
        if length(data_unused) > 0
            data = vcat(data_unused, newdata)
            data_unused = Uint16[]
        else
            data = copy(newdata)
        end
        deblip_nsls_data!(data, -8)

        # Select times from the pulse_times list
        t1 = t2
        while t2 <= np && pulse_times[t2] < nsamp_read
            t2 += 1
        end
        t2 <= t1 && continue  # There were no pulses to fit for

        # Is t2 too close to the end of the vector? If so, back off and save some
        # data for later. Remove one (or more) triggers from the trigger set AND
        # trim down the data vector.
        # But also be careful: if you trim TOO much, then the next record could
        # become unboundedly large. We will stop trimming when the length of this
        # becomes less than one half of a normal chunk.
        last_samp = nsamp_read
        while last_samp < pulse_times[t2-1] + MIN_POST_TRIG_SAMPS
            cut_out_samples = last_samp - (pulse_times[t2-1] - MIN_PRE_TRIG_SAMPS)
            last_samp -= cut_out_samples
            if cut_out_samples < length(data)
                data_unused = vcat(data[end+1-cut_out_samples:end], data_unused)
                data = data[1:end-cut_out_samples]
            else
                data_unused = vcat(data, data_unused)
            end
            t2 -= 1
            t2 == t1 && break  # No more pulses in the interval!
            if length(data_unused) >= div(file.nsamp*nrecs, 2) # Too much was trimmed
                while last_samp < pulse_times[t2-1]
                    cut_out_samples = last_samp - (pulse_times[t2-1] - MIN_POST_TRIG_SAMPS)
                    if cut_out_samples >= length(data)
                        cut_out_samples = length(data)-1
                    end
                    last_samp -= cut_out_samples
                    data_unused = vcat(data[end-cut_out_samples:end], data_unused)
                    data = data[1:end-cut_out_samples]
                    t2 -= 1
                    t2==t1 && break  # No more pulses in the interval!
                end
                break
            end
            #if (length(data_unused) + 2*length(newdata) > 300000) # MAXSAMP
            #    break
            #end
        end

        ncp = t2-t1
        ncp <= 0 && continue  # There were no pulses to fit for

        const TRIG_TIME_OFFSET = -2
        current_times = pulse_times[t1:t2-1] - (last_samp-length(data)) + TRIG_TIME_OFFSET
        if current_times[end] >= length(data)
            warn("Problem!")
            @show current_times, length(data)
        end

        floatdata = float(data)
        N = min(length(tails), length(data))
        floatdata[1:N] -= tails[1:N]
        if DO_WITH_PLOTS
            @show pulse_times[t1:t2-1]
            param, covar, this_residW, this_residR, this_prior = 
                multi_pulse_fit_with_plot(floatdata, current_times,
                                          mpf, use_dpdt)
            println("That was segment $i\n")
            i>=37 && error("End of test on section $(i)!")
        else
            param, covar, this_residW, this_residR, this_prior = 
                multi_pulse_fit(floatdata, current_times,
                                mpf, use_dpdt)
        end

        # Figure out the tail you'd expect in the next record
        const NTAIL = 600
        tails = zeros(Float64, NTAIL)
        for i = 1:length(current_times)
            cti = length(data) - current_times[i]
            if NTAIL-cti <= 0 || cti <= 0
                continue
            end
            tails[1:NTAIL-cti] += param[i]*mpf.pulse_model[1+cti:NTAIL]
        end

        
        pheights[np_seen+1:np_seen+ncp] = param[1:ncp]
        pheight_unc[np_seen+1:np_seen+ncp] = sqrt(diag(covar)[1:ncp])
        if use_dpdt
            dpdts[np_seen+1:np_seen+ncp] = param[1+ncp:2*ncp]
            dpdt_unc[np_seen+1:np_seen+ncp] = sqrt(diag(covar)[1+ncp:2*ncp])
            baseline[np_seen+1:np_seen+ncp] = param[2*ncp+1]
        else
            baseline[np_seen+1:np_seen+ncp] = param[ncp+1]
        end
        prior_level[np_seen+1:np_seen+ncp] = this_prior
        residW[np_seen+1:np_seen+ncp] = this_residW
        residR[np_seen+1:np_seen+ncp] = this_residR
        np_seen += ncp
        if mod(i,1000) == 0
            println("$i chunks seen, with $(np_seen) pulses fit.")
        end
    end
    pheights, dpdts, pheight_unc, dpdt_unc, prior_level, residW, residR, baseline
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

