"""
Multi-pulse fitting driver.

Joe Fowler, Autumn 2014

"""

# Key parameters to control the multi-pulse fitting
BG_FIT_SLOPE = false
DO_WITH_PLOTS =  false

using HDF5, PyPlot, Dierckx
cd("/Users/fowlerj/Software/mass/julia")
using MicrocalFiles, HDF5Helpers, CovarianceModels

include("NoiseAnalysis.jl")
include("RandomMatrix.jl")

# While developing, use the following in place of "using"
# include("CovarianceModels.jl")
# choleskysolve = CovarianceModels.choleskysolve
# CovarianceModel = CovarianceModels.CovarianceModel
# StreamCovarianceSolver = CovarianceModels.StreamCovarianceSolver
# BadCovarianceSolver = CovarianceModels.BadCovarianceSolver
# ExactCovarianceSolver = CovarianceModels.ExactCovarianceSolver



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


function MPF_analysis_2011(setname::String="C_auto", noiseset::String="C",
                           pulseshape=nothing, dpdt=nothing)
    const PATH="/Volumes/Data2014/Data/Data_CDM/2011_09_12"
    const cnum = 1
    filename = @sprintf("%s/2011_09_12_%s_chan%d.ljh", PATH, setname, cnum)
    noisename = @sprintf("%s/2011_09_12_%s_chan%d.noi", PATH, noiseset, cnum)
    println("Working on $filename noise")
    hdf5name = hdf5_name_from_ljh_name(filename, "mpf")
    Noise_analysis(hdf5name, noisename)
    println("Working on $filename data")

    # If given a pulse model, store that in the HDF5 file.
    if pulseshape != nothing

        # If not given dpdt, estimate from finite difference
        if dpdt == nothing
            dpdt = finite_diff_deriv(pulseshape)
        end
        println("Storing new pulse to HDF5")

        h5file = h5file_update(hdf5name)
        g = g_create_or_open(h5file, "chan1")
        ds_update(g, "average_pulse", pulseshape)
        ds_update(g, "average_dpdt", dpdt)
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

function load_model_pulse(modelname::String)

    model = Dict{String,Vector{Float64}}()

    # Experiment using my 2013 model pulse:
    t=[0:100000]
    model["model2013"] = (-5807.6803*(.980251.^t)-23588.2151*(.99033.^t)
                          +37401.5698*(.999439.^t)+423.2109*(.999805.^t)
                          +(-8518.9679*cos(.00033*t)
                            -11750*sin(.00033*t)).*exp(-t/910.3435))
    model["model2014"] = (-11074.9904*(.983966.^t) -19066.1923*(0.991594.^t)
                          + 31235.6323*(0.999480.^t) + (.998601.^t).*
                          (-1196.5864 *cos(0.00079* t)-4785.0870 *sin(0.00079* t)));

    # Load the empirical one
    h5file = h5open("/Volumes/Data2014/Data/Data_CDM/2011_09_12/2011_09_12_mpf_pulse.hdf5",
                    "r")
    model["average_pulse"] = read(h5file["model_pulse"])
    close(h5file)

    model["spline"] = spline_model_pulse()

    if haskey(model, modelname)
        return model[modelname]
    else
        error("Model names allowed are $(keys(model)).")
    end
end


function spline_model_pulse()
    # Load the 2013 spline model
    npzdata=h5open("/Users/fowlerj/Microcal/Continuous_runs/modelpulse_20110912.hdf5","r")
    try
        saved_empirical = read(npzdata["pulse_shape"])
    finally
        close(npzdata)
    end
    const NSHORT=5000
    const NTOTAL=100000
    t = float([0:NSHORT])
    saved_empirical = vcat([0.0], saved_empirical)
    spl = Spline1D( t, saved_empirical, s=NSHORT, w=4.0 ./ log10(10+t) )
    model = Array(Float64, NTOTAL)
    model[1:NSHORT+1] = evaluate(spl, t)

    tlater = float([0:NTOTAL-NSHORT-2])
    a1, a2 = 2201.4096, 77.06
    model[NSHORT+2:end] = (a1*exp(-tlater*.0005381353) +
                           a2*exp(-tlater*.0001271194))
    model
end


function copy_noise_to_all_2011()
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
end


function oct27_work()
    pulse_shape = load_model_pulse("model2014");
    #pulse_shape = load_model_pulse("model2013");
    #pulse_shape = load_model_pulse("average_pulse");

    #@time MPF_analysis_2011("C_auto", "C", pulse_shape)
    # copy_noise_to_all_2011()

    @time MPF_analysis_2011("D_auto", "C", pulse_shape)
    @time MPF_analysis_2011("E_auto", "C", pulse_shape)
    @time MPF_analysis_2011("F_auto", "C", pulse_shape)
    @time MPF_analysis_2011("G_auto", "C", pulse_shape)
    @time MPF_analysis_2011("H_auto", "C", pulse_shape)
    #@time MPF_analysis_2011("I_auto", "C", pulse_shape)
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
            nlags, nexps = 2500, 2 # For 2011 TDM data
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
        const do_all_trigtimes = false
        const do_avg_pulse = false
        const do_all_fits = true

        grpname=string("chan$(file.channum)")
        h5grp = g_create_or_open(h5file, grpname)

        # Find trigger times
        if forceNew || ! exists(h5grp, "trigger_times") | do_all_trigtimes
            # NSLS trig_times = find_all_pulses(file)
            trig_times = find_all_pulses(file, 65)
            ds_update(h5grp, "trigger_times", trig_times)
        end
        trigger_times = read(h5grp["trigger_times"])

        # Analyze the average pulse
        if forceNew || ! exists(h5grp, "average_pulse") | do_avg_pulse
            println("\nComputing average pulse shape...")
            avg_pulse = compute_average_pulse(file, trigger_times)
            dpdt = finite_diff_deriv(avg_pulse)
            ds_update(h5grp, "average_pulse", avg_pulse)
            ds_update(h5grp, "average_dpdt", dpdt)
        end

        # Do the fits
        if forceNew || ! exists(h5grp, "mpf") | do_all_fits
            avg_pulse = read(h5grp["average_pulse"])
            dpdt = read(h5grp["average_dpdt"])
            MAXSAMP = 400000  # 8000 for NSLS
#            covar_model = ExactCovarianceSolver(read_complex(h5grp["noise/model_amplitudes"]),
#                                          read_complex(h5grp["noise/model_bases"]),
#                                          MAXSAMP)

            # Try the model from lab book 6 page 124.
            covar_model = ExactCovarianceSolver([82.5463,24.8891,13.6768,3.1515,0],
                                          [.0871940295,.9941810695,
                                           .9992481192,.9999235960], MAXSAMP)

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
            threshold = 6.5*mad # NSLS 6*mad
            clf()
            PyPlot.plt.hist(x[5:end], 1000, histtype="step", color="r")
            println( threshold)
        end

        skip_until = 0
        for i in 2:Ndata
            i<skip_until && continue
            if x[i]>threshold && d[i] > 1500# 50+data_med
                push!(trigger_times, i+records_read*file.nsamp)
                skip_until = i+10
            end
        end

        history = d[end-4:end]
        records_read += nrec

        print("Read $records_read of $(file.nrec) data records for trigger analysis.  ")
        println("Found $(length(trigger_times)) so far.")
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
        if badness[i] > critical_badness ||
            pulses[288+PRE_PERIOD,i] < 20000 || pulses[288+PRE_PERIOD,i] > 30000 # MnKA
            continue
        end
        avg_pulse += pulses[:, i]
        np += 1
    end

    println("Computed avg pulse from $np clean pulses out of $(length(isolated_times)) isolated.")
    avg_pulse = avg_pulse / float(np)
    avg_pulse -= mean(avg_pulse[1:PRE_PERIOD-3])
    avg_pulse = avg_pulse[PRE_PERIOD-2:end]

    # For 2011_09_12 data, do the following to extend the data:
    t = [1:100000]
    avg_pulse = vcat( avg_pulse[1:5000-2], 1.002278815*(
                     2243.93*exp(-t/1862.65) + 73.84*exp(-t/9264.97)))
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
    const MAXLEN = 100000 # NSLS was 5120
    covar_model = ExactCovarianceSolver(read_complex(noisegrp["model_amplitudes"]),
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
    if typeof(mpf) == ExactCovarianceSolver && chunklength > mpf.covar_model.max_length
        @show chunklength, mpf.covar_model.max_length, length(mpf.white_const)
        error("Cannot extend the MultiPulseFitter longer than the ExactCovarianceSolver")
    end

    # Whiten baseline functions to their new length
    mpf.white_slope = choleskysolve(mpf.covar_model, linspace(-1, 1, chunklength))
    mpf.white_const = choleskysolve(mpf.covar_model, ones(Float64, chunklength))
#      clf(); plot(mpf.white_const,"r", mpf.white_slope,"g"); plot(yomama)
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
            choleskysolve(mpf.covar_model, mpf.pulse_model[1:pend-pstart+1],
                                pstart-1)[pstart:end]
        if use_dpdt
            model_components[pstart:pend, i+Np] =
            choleskysolve(mpf.covar_model, mpf.dpdt_model[1:pend-pstart+1],
                                    pstart-1)[pstart:end]
        end
    end

    # Compute the model's "design matrix".
    # The following BLAS call seemed like it might be faster, but it was no different.
    # A = BLAS.gemm('T', 'N', model_components, model_components)
    A = model_components' * model_components
    covar = inv(A)

    white_data = choleskysolve(mpf.covar_model, data)
#     clf(); plot(model_components); plot(white_data/5000,color="brown"); plot(yomama)

    mc_dot_data = (model_components' * white_data)
    param = A \ mc_dot_data
    model = model_components * param
    #unwhite_model = choleskyproduct(mpf.covar_model, model, length(model))
    if BG_FIT_SLOPE
        unwhite_model = ones(Float64, Nd) * param[end-1]
        unwhite_model += linspace(-1, 1, Nd)
    else
        unwhite_model = ones(Float64, Nd) * param[end]
    end
    for i = 1:Np
        pstart = pulse_times[i]
        pend = min(pstart+length(mpf.pulse_model)-1, Nd)
        pstart = max(1, pstart)
        unwhite_model[Nd-pend+pstart:end] += mpf.pulse_model[1:pend-pstart+1]*param[i]
        if use_dpdt
            unwhite_model[Nd-pend+pstart:end] += mpf.dpdt_model[1:pend-pstart+1]*param[i+Np]
        end
    end
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
    subplot(313)
    plot(model, color="k")
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
    subplot(311)
    plot(data, "r", model, "b"); title("Data (red) and model (blue)")
    subplot(312)
    plot((data-model), "g"); title("Residuals")
    param, covar, residW, residR, prior_level
end



# MPF action on an entire file.

function multi_pulse_fit_file(file::MicrocalFiles.LJHFile, pulse_times::Vector,
                         mpf::MultiPulseFitter, use_dpdt::Bool=false)
    const nrecs = 1
    extendMPF!(mpf, (nrecs+1)*file.nsamp)

    const MIN_POST_TRIG_SAMPS = 15000
    const MIN_PRE_TRIG_SAMPS = 5000
    # const MIN_POST_TRIG_SAMPS = 200 NSLS
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
    tails = zeros(Float64, 70000)

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
            i>=6 && error("End of test on section $(i)!")
        else
            param, covar, this_residW, this_residR, this_prior =
                multi_pulse_fit(floatdata, current_times,
                                mpf, use_dpdt)
        end

        # Figure out the tail you'd expect in the next record
        const NTAIL = 70000
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


# Investigate statistical power
function explore_power(mpf::MultiPulseFitter)
    clf()
    const ND=3000
    for s0 in 50:100:ND
        s = zeros(Float64, ND)
        s[s0:end] = mpf.pulse_model[1:ND+1-s0]
        swhite = CovarianceModels.whiten(mpf.covar_model, s)
        M=hcat(swhite, mpf.white_const[1:ND])
        A=inv(M'*M)
        dph = sqrt(A[1,1])
        #        dph=1/sqrt(norm(swhite))
        dbl = sqrt(A[2,2])
        plot(s0, dph, "ro")
        plot(s0, dbl, "bo")
        s0==2450 && @show(s0, dph, dbl)
    end
end

function explore_power2(mpf::MultiPulseFitter)
    ND1=12000
    ND2=24000

    s1=zeros(Float64,ND1)
    s2=zeros(Float64,ND2)
    s3a=zeros(Float64,ND2)
    s3b=zeros(Float64,ND2)
    s1[2001:ND1] = mpf.pulse_model[1:ND1-2000]
    s2[2001:ND2] = mpf.pulse_model[1:ND2-2000]
    s3a[2001:ND2] = mpf.pulse_model[1:ND2-2000]
    s3b[12001:ND2] = mpf.pulse_model[1:ND2-12000]
    clf()
    t=[1-2000:ND2-2000]*.00064
    plot(t[1:ND1], s1, "g", t, s2+.4,"b", t, s3a+s3b+.8,"r")
    xlabel("Time (ms) after first pulse")
    ylabel("Pulse (arbs)")

    s1white = CovarianceModels.whiten(mpf.covar_model, s1)
    s2white = CovarianceModels.whiten(mpf.covar_model, s2)
    s3awhite = CovarianceModels.whiten(mpf.covar_model, s3a)
    s3bwhite = CovarianceModels.whiten(mpf.covar_model, s3b)
    M1=hcat(s1white, mpf.white_const[1:ND1])
    M2=hcat(s2white, mpf.white_const[1:ND2])
    M3=hcat(s3awhite, s3bwhite, mpf.white_const[1:ND2])
    A1=inv(M1'*M1)
    A2=inv(M2'*M2)
    A3=inv(M3'*M3)

    dph1 = sqrt(A1[1,1])
    dbl1 = sqrt(A1[2,2])
    dph2 = sqrt(A2[1,1])
    dbl2 = sqrt(A2[2,2])
    dph3a = sqrt(A3[1,1])
    dph3b = sqrt(A3[2,2])
    dbl3 = sqrt(A3[3,3])
    @show dph1, dph2, dph3a, dph3b
    @show dbl1, dbl2, dbl3
    @show dph2/dph1, dph3a/dph1, dph3b/dph1
    nothing
end


function filter_expected()
    # Try the model from lab book 6 page 124.
    const MAXSAMP = 40000
    covar_model = CovarianceModel([82.5463,24.8891,13.6768,3.1515,0],
                                  [.0871940295,.9941810695,
                                   .9992481192,.9999235960], MAXSAMP)
    pulse_model = load_model_pulse("model2014")
    const SCALE = 5898*2.3548

    clf()
    N=vcat([400:200:1000],
           [1500:500:10000],
           [11000:1000:20000],
           [22000:2000:40000])
    n=length(N)
    de=zeros(Float64, n, 5)

    for j=1:length(N)
        n = N[j]
        white_const = CovarianceModels.whiten(covar_model, ones(Float64,n))
        for i in 1:4
            m = div(n*i,5)
            model= zeros(Float64,n)
            model[1+n-m:end] = pulse_model[1:m]
            white_model = CovarianceModels.whiten(covar_model, model)
            white_dpdt = CovarianceModels.whiten(covar_model, finite_diff_deriv(model))
            M=hcat(white_model,white_const, white_dpdt)
            A = M'*M
            de[j,i] = SCALE*sqrt(inv(A)[1,1])
        end
        m = div(n*4,5)
        model= zeros(Float64,n)
        model[1+n-m:end] = pulse_model[1:m]
        white_model = CovarianceModels.whiten(covar_model, model)
        white_dpdt = CovarianceModels.whiten(covar_model, finite_diff_deriv(model))
        expterms = exp([1:n].*0.000538), exp([1:n] .* 0.00012712)
        white_exp1 = CovarianceModels.whiten(covar_model, expterms[1])
        white_exp2 = CovarianceModels.whiten(covar_model, expterms[2])
        M=hcat(white_model, white_const, white_exp1,  white_exp2, white_dpdt)
        A = M'*M
        de[j,5] = SCALE*sqrt(inv(A)[1,1])
    end
    plot_filter_expected(N, de, true)
    N, de
end


function plot_filter_expected(N::Vector{Int}, de::Matrix{Float64},
                              vsN::Bool)
    clf()
    const DT=640e-9
    const SAMP_PER_MS=1/.00064
    #rate_exps2 = exp(-1.0)/DT ./ (N+4*SAMP_PER_MS)
    rate_exps1 = exp(-1.0)/DT ./ (N+4*SAMP_PER_MS)
    rate_exps2 = exp(-1.0)/DT ./ (N+2*SAMP_PER_MS)
    rate_std = exp(-1.0)/DT ./ (N+8*SAMP_PER_MS)
    if vsN; x=N.* (DT*1000); else x=rate_std; end
    plot(x, de[:,1], "-og", label="80% pretrigger")
    plot(x, de[:,2], "-oc", label="60% pretrigger")
    plot(x, de[:,3], "-ob", label="40% pretrigger")
    plot(x, de[:,4], "-o", color="purple", label="20% pretrigger")

    if vsN;
        x= N .* (DT*1000)
        plot(x, de[:,5], "-o", label="20% PT, Xexps", color="orange")
        legend(loc="upper right")
    else
        x=rate_exps1
        plot(x, de[:,5], "-o", label="20% PT, Xexps, 4 ms veto", color="orange")
        plot(x, de[:,4], "-o", label="20 PT, 4 ms veto", color="r")
        #x=rate_exps2
        #plot(x, de[:,5], "--o", label="20% PT, Xexps, 2 ms veto", color="#cc8844")
        #plot(x, de[:,4], "--o", label="20 PT, 2 ms veto", color="#cc4444")
        plot(zeros(Float64,2)+1000./(8*exp(1)), [2.3,4], "--", color="gray")
        legend(loc="upper left")
    end
    ylabel("Theoretical best energy resolution (eV)")
    if vsN
        xlim([.3, 30])
        semilogx()
        xt = [0.3, 0.5, 1, 2, 5, 10, 20]
        plt.xticks(xt, [string(t) for t in xt])
        xlabel("Length of filter (ms)")
    else
        xlim([10,90])
        xlabel("Theoretical maximum rate (Hz)")
    end
    ylim([2.3,4])
    title("Resolution rate trade-off, assuming NO violation of model assumptions")
end



# For comparison, traditional filtering
function assess_traditional_filters(noise_model::Union(Bool, NoiseSummary)=false)
    const MAXSAMP = 32000
    covar_model = CovarianceModel([82.5463,24.8891,13.6768,3.1515,0],
                                  [.0871940295,.9941810695,
                                   .9992481192,.9999235960], MAXSAMP)

    # Convert true covariance model from a CovarianceModel to a toeplitz matrix
    if noise_model == false
        true_covar_model = covar_model
    else
        true_covar_model = CovarianceModel(noise_model.exp_amplitudes,
                                     noise_model.exp_bases, MAXSAMP)
    end
    true_acorr = zeros(Float64, MAXSAMP) + real(true_covar_model.bases[end])
    t = float([0:MAXSAMP-1])
    for i=1:true_covar_model.num_exps-1
        true_acorr += real(true_covar_model.amplitudes[i] * (true_covar_model.bases[i].^t))
    end

    pulse_model = load_model_pulse("model2014")
    dpdt_model = finite_diff_deriv(pulse_model)
    const SCALE = 5898*2.3548

    filterN = [31248, 7812, 3125, 1562, 1562, 1562, 1562, 7812, 3125]
    filtNPT = [16000, 4687, 1875,  312,  625,  937, 1250, 4687, 1875]
    names = ["20ms", "5ms", "2ms", "1ms20", "1ms40", "1ms60", "1ms80", "5msexp", "2msexp"]

    for i = 1:length(filterN)
        npre = filtNPT[i]
        nsamp = filterN[i]
        npost = nsamp-npre
        ncol = i<=6 ? 3 : 5

        model = zeros(Float64, nsamp, ncol)
        modelW = zeros(Float64, nsamp, ncol)
        model[npre+1:end, 1] = pulse_model[1:npost]
        model[npre+1:end, 2] = dpdt_model[1:npost]
        model[:, 3] = 1.0
        if ncol > 3
            t = float([1:nsamp])
            model[:, 4] = exp(-t ./ 1858.74)
            model[:, 5] = exp(-t ./ 7866.58)
        end

        for j = 1:ncol
            modelW[:,j] = CovarianceModels.whiten(covar_model, model[:,j])
        end
        A = modelW' * modelW

        RinvM = Array(Float64, nsamp, ncol)
        for c =1:ncol
            RinvM[:,c] = CovarianceModels.covarsolve(covar_model, model[:,c])
        end

        # Can't afford to construct the Toeplitz T, the true noise
        # But we can multiply T * RinvM
        #TRinvM = Array(Float64, nsamp, ncol)
        #for c=1:ncol
            #TRinvM[:,c] = reverse(xcorr(true_acorr[1:nsamp], RinvM[:,c])[1:nsamp])
            #TRinvM[2:end,c] += xcorr(true_acorr[2:nsamp], RinvM[:,c])[1:nsamp-1]
            # TRinvM[1,c] = dot(true_acorr[1:nsamp], RinvM[:,c])
            # TRinvM[end,c] = dot(rev_true_acorr[1+end-nsamp:end], RinvM[:,c])
            # for r=2:nsamp-1
            #     TRinvM[r,c] = (dot(true_acorr[1:nsamp+1-r], RinvM[r:end,c]) +
            #                    dot(rev_true_acorr[end-r+1:end-1], RinvM[1:r-1,c]))
            # end
        #end
        r = copy(true_acorr[1:nsamp])
        r[1] *= 0.5
        X = Array(Float64, nsamp, ncol)
        for c=1:ncol
            X[:,c] = xcorr(RinvM[:,c], r)[end+1-nsamp:end]
        end
        MTRiTRiM = RinvM' * X
        MTRiTRiM += MTRiTRiM'


        Cpp = inv(A) * MTRiTRiM * inv(A)
        println(sqrt(Cpp[1,1])*SCALE)
    end
end



function create_traditional_filters(noise_model::Union(Bool, NoiseSummary)=false,
                                    long2015::Bool=false)
    # Use long2015=true for the final 2015 analysis with longer filters
    # Use long2015=false for the Nov 2014 analysis to study effect of varying
    # filter length AND pretrigger/posttrigger fractions.

    const MAXSAMP = 40000
    if noise_model == false
        covar_model = ExactCovarianceSolver([82.5463,24.8891,13.6768,3.1515,0],
                                      [.0871940295,.9941810695,
                                       .9992481192,.9999235960], MAXSAMP)
    else
        covar_model = ExactCovarianceSolver(noise_model.exp_amplitudes,
                                      noise_model.exp_bases, MAXSAMP)
    end
    pulse_model = load_model_pulse("model2014")
    #pulse_model /= maximum(pulse_model)*.1
    dpdt_model = finite_diff_deriv(pulse_model)
    const SCALE = 5898*2.3548

    if long2015
        filterN = [31248, 15624, 7812, 3906, 7812, 3906]
        filtNPT = [18748,  9374, 4687, 2344, 4687, 2344]
        names = ["20ms", "10ms", "5ms", "2.5ms", "5msexp", "2.5msexp"]
    else
        filterN = [7812, 3125, 1562, 1562, 1562, 1562, 7812, 3125]
        filtNPT = [4687, 1875,  312,  625,  937, 1250, 4687, 1875]
        names = ["5ms", "2ms", "1ms20", "1ms40", "1ms60", "1ms80", "5msexp", "2msexp"]
    end

    filters = Dict{String,Vector{Float64}}()
    clf()
    for i = 1:length(names)
        npre = filtNPT[i]
        nsamp = filterN[i]
        npost = nsamp-npre
        ncol = 3
        if (long2015 && i>=5) || i >= 7
            ncol = 5
        end

        model = zeros(Float64, nsamp, ncol)
        modelW = zeros(Float64, nsamp, ncol)
        model[npre+1:end, 1] = pulse_model[1:npost]
        model[npre+1:end, 2] = dpdt_model[1:npost]
        model[:, 3] = 1.0
        if ncol > 3
            t = float([1:nsamp])
            model[:, 4] = exp(-t ./ 1858.74)
            model[:, 5] = exp(-t ./ 7866.58)
        end

        for j = 1:ncol
            modelW[:,j] = CovarianceModels.whiten(covar_model, model[:,j])
        end
        A = modelW' * modelW
        RinvM = Array(Float64, nsamp, ncol)
        for c =1:ncol
            RinvM[:,c] = CovarianceModels.covarsolve(covar_model, model[:,c])
        end
        allfilt = A \ (RinvM')
        println(sqrt(inv(A)[1,1])*SCALE)
        filters[names[i]] = vec(allfilt[1,:])
        x = .00064 .* [1-npre:npost]
        if i==1
            plot(.00064 .* [0:npost], .8+pulse_model[1:npost+1] .* 7e-6, "k", label="Pulse")
            plot(.00064 .* [1-npre:0], .8+zeros(Int,npre), "k")
        end
        plot(x, .8-i*0.08+filters[names[i]].*1e5, label=names[i])
    end
    legend(loc="center left")

    npre = Dict{String,Int}()
    for i=1:length(names)
        npre[names[i]] = filtNPT[i]
    end
    filters, npre
end



function classic_analysis(filename::String, forceNew::Bool=false,
                          long2015::Bool=false)
    file = MicrocalFiles.LJHFile(filename)
    hdf5name = hdf5_name_from_ljh_name(filename, "mpf")
    println("We are about to filter data; store into\n$hdf5name")
    h5file = h5file_update(hdf5name)
    try
        const do_all_trigtimes = false
        const do_all_fits = true

        grpname=string("chan$(file.channum)")
        h5grp = g_create_or_open(h5file, grpname)

        # Find trigger times
        if forceNew || ! exists(h5grp, "trigger_times") | do_all_trigtimes
            # NSLS trig_times = find_all_pulses(file)
            trig_times = find_all_pulses(file, 65)
            ds_update(h5grp, "trigger_times", trig_times)
        end
        trigger_times = read(h5grp["trigger_times"])

        # Do the fits
        if forceNew || ! exists(h5grp, "classic") | do_all_fits
            MAXSAMP = 10000

            println("Computing classic pulse filters...\n")
            filters, npre = create_traditional_filters(false, long2015)

            println("Computing classic pulse fits...\n")
            results = filter_file(file, trigger_times, filters, npre)

            grp = g_create_or_open(h5grp, "classic")
            for k in keys(results)
                ds_update(grp, k, results[k])
            end
        end
    finally
        close(h5file)
    end
    nothing
end


function filter_file(file::LJHFile,
                     trigger_times::Vector{Int},
                     filters::Dict{String,Vector{Float64}},
                     npre::Dict{String,Int})

    const TRIG_TIME_OFFSET = -2
    const np = length(trigger_times)
    results = Dict{String, Vector{Float64}}()
    for k in keys(filters)
        results[k] = Array(Float64, np)
    end

    const nrecs = 6
    MicrocalFiles.LJHRewind(file)
    _times = Array(Uint64, nrecs)
    newdata = Array(Uint16, file.nsamp*nrecs)
    data_unused = Uint16[]
    t1,t2 = 0,1
    nsamp_read = 0
    np_seen = 0

    const MIN_POST_TRIG_SAMPS = 13000
    const MIN_PRE_TRIG_SAMPS = 20000

    for i=1:div(file.nrec, nrecs)
        MicrocalFiles.fileRecords(file, nrecs, _times, newdata)
        nsamp_read += length(newdata)
        if length(data_unused) > 0
            data = vcat(data_unused, newdata)
            data_unused = Uint16[]
        else
            data = copy(newdata)
        end
        # If record is too short, add the next one
        if length(data) < MIN_PRE_TRIG_SAMPS + MIN_POST_TRIG_SAMPS
            data_unused = data
            continue
        end

        # Select times from the trigger_times list
        first_read = nsamp_read - length(data)
        while (t2 < length(trigger_times) &&
               trigger_times[t2] - 2 < first_read)
            t2 += 1
        end
        t1 = t2
        while t2 <= np && trigger_times[t2] < nsamp_read
            t2 += 1
        end
        t2 <= t1 && continue # No pulses to fit for

        # Is t2 too close to the end of the vector? If so, back off and save some
        # data for later. Remove one (or more) triggers from the trigger set AND
        # trim down the data vector.
        # But also be careful: if you trim TOO much, then the next record could
        # become unboundedly large. We will stop trimming when the length of this
        # becomes less than one half of a normal chunk.
        last_samp = nsamp_read
        while last_samp < trigger_times[t2-1] + MIN_POST_TRIG_SAMPS
            cut_out_samples = last_samp - (trigger_times[t2-1] - MIN_PRE_TRIG_SAMPS)
            last_samp -= cut_out_samples
            if cut_out_samples < length(data)
                data_unused = vcat(data[end+1-cut_out_samples:end], data_unused)
                data = data[1:end-cut_out_samples]
            else
                data_unused = vcat(data, data_unused)
            end
            t2 -= 1
            t2 == t1 && break  # No more pulses in the interval!
        end

        ncp = t2-t1

        if ncp > 0
            const TRIG_TIME_OFFSET = -2
            current_times = trigger_times[t1:t2-1] - (last_samp-length(data)) + TRIG_TIME_OFFSET
            if current_times[end] >= length(data)
                warn("Problem!")
                @show current_times, length(data)
            end

            floatdata = float(data)

            # Do the filters!
            for k in keys(filters)
                f = filters[k]
                pre = npre[k]
                nsamp = length(f)
                post = nsamp - pre
                theseresults = results[k]

                for j=1:length(current_times)
                    npulse = j-1+t1
                    t = current_times[j]
                    if t<=pre; continue; end
                    theseresults[npulse] = dot(f,  floatdata[t-pre:t-pre-1+nsamp])
                end
            end
        end

        # Save last 5k samples for use next time
        samp2save = min(5000, length(data))
        data_unused = vcat(data[end+1-samp2save:end], data_unused)

        np_seen += ncp
        if mod(i,1000) == 0
            println("$i chunks seen, with $(np_seen) pulses fit.")
        end
    end

    #####

    results
end

expand_name(setname) =
    @sprintf("/Volumes/Data2014/Data/Data_CDM/2011_09_12/2011_09_12_%s_chan1.ljh",
             setname)

function nov25_work()
    @time classic_analysis( expand_name("C_auto"))
    @time classic_analysis( expand_name("D_auto"))
    @time classic_analysis( expand_name("E_auto"))
    @time classic_analysis( expand_name("F_auto"))
    @time classic_analysis( expand_name("G_auto"))
end

function feb6_2015_work()
    @time classic_analysis( expand_name("C_auto"), false, true)
    @time classic_analysis( expand_name("D_auto"), false, true)
    @time classic_analysis( expand_name("E_auto"), false, true)
    @time classic_analysis( expand_name("F_auto"), false, true)
    #@time classic_analysis( expand_name("G_auto"), false, true)
end



function find_noise_level(setname::String, nrecords::Int=20)
    ljhname = "/Volumes/Data2014/Data/Data_CDM/2011_09_12/2011_09_12_$(setname)_chan1.ljh"
    f = MicrocalFiles.LJHFile(ljhname)
    nrecords = min(f.nrec, nrecords)
    _times = Array(Uint64, nrecords)
    data = Array(Uint16, nrecords*f.nsamp)
    MicrocalFiles.fileRecords(f, nrecords, _times, data)

    hdf5name = hdf5_name_from_ljh_name(ljhname, "mpf")
    println("We are about to compute MPF results; store into\n$hdf5name")
    h5file = h5open(hdf5name, "r")
    ttimes = read(h5file,"chan1/trigger_times")
    close(h5file)

    const SEGLENGTH=32768*2    # ~42 ms
    const WAITAFTERPULSE=25000 # 16 ms
    const BADBEFOREPULSE=100
    segstarts = Int[]

    ttimes = ttimes[ttimes .< length(data)-BADBEFOREPULSE]
    for it = 2:length(ttimes)
        tt=ttimes[it]
        nsegs = div(tt-ttimes[it-1]-WAITAFTERPULSE-BADBEFOREPULSE, SEGLENGTH)
        nsegs < 1 && continue
        se = tt - BADBEFOREPULSE - nsegs*SEGLENGTH
        for j=1:nsegs
            thisr = se:se+SEGLENGTH-1
            drange = convert(Int64, maximum(data[thisr])-minimum(data[thisr]))
            if drange< 140
                push!(segstarts, se)
            else
                println("Ignoring segment starting at $(se) for data range $(drange)")
            end
            se += SEGLENGTH
        end
    end

    nsegs = length(segstarts)
    println("Averaging $(nsegs) together")
    zeropad = zeros(Float64, SEGLENGTH*2)
    autocorr = zeros(Float64, SEGLENGTH)
    clf()
    t = .00064*float([0:SEGLENGTH-1])
    for j=1:nsegs
        thisr = segstarts[j]:segstarts[j]+SEGLENGTH-1
        zeropad[1:SEGLENGTH] = data[thisr] .- mean(data[thisr])
        dft = fft(zeropad)
        psd = dft .* conj(dft)
        psd[1] = 0
        ac = real(ifft(psd))[1:SEGLENGTH] ./ SEGLENGTH
        autocorr += ac
        #plot(100*j+zeropad[1:SEGLENGTH])
        plot(t,ac)
    end
    autocorr /= nsegs
    plot(t,autocorr, "k", lw=3)
    autocorr
end


function assess_noise()
    acorr = Dict{Char,Vector{Float64}}()
    keys="CDEFG"
    nrecs=[1000,1000,5000,10000,30000]
    for i=1:5
        key=keys[i]
        acorr[key] = find_noise_level("$(key)_auto", nrecs[i])
    end
    plot_noise(acorr)
    acorr
end

function plot_noise(acorr::Dict{Char,Vector{Float64}},
                    plotpsd::Bool=false)
    clf()
    keys="CDEFG"
    colors=["purple","b","g","gold","r"]
    if plotpsd
        nf=div(length(acorr['C']), 2)
        t = float([0:nf-1]) *0.5/(nf*640e-9)
        df = t[2]-t[1]
        xlabel("Frequency (Hz)")
        title("Noise PSD (arb^2 / Hz) from quiet periods")
    else
        t=.00064*float([1:24999])
        xlabel("Lag (ms)")
        title("Noise autocorrelation from quiet periods")
    end
    for i=1:5
        key=keys[i]
        if plotpsd
            psd = fft(acorr[key])
            psd = abs(psd).^2
            loglog(t, psd[1:nf] ./ df, color=colors[i], label="Set $(key)")
        else
            plot(t, acorr[key][2:25000], color=colors[i], label="Set $(key)")
            plot(0, acorr[key][1], "o", color=colors[i])
        end
    end
    legend(loc="upper right")
end



#####
# Demonstration of the method: make a plot using NSLS data
# This becomes paper figure 2.
#####

function demoplot()
    cnum = 119
    setname = "T"
    date = "14"
    const PATH="/Volumes/Data2014/Data/NSLS_data/2012_06_$(date)"
    filename = @sprintf("%s/2012_06_%s_%s_chan%d.ljh", PATH, date, setname, cnum)

    ljhfile = MicrocalFiles.LJHFile(filename)
    nrecs=20
    _times = Array(Uint64, nrecs)
    data = Array(Uint16, ljhfile.nsamp, nrecs)
    MicrocalFiles.fileRecords(ljhfile, nrecs, _times, data)
    data = reshape(data, length(data))
    deblip_nsls_data!(data, -9)

    filter = [.4, .1, -.2, -.5, -.8, 1]
    trigger = xcorr(data, filter)[length(data)-length(filter)-1:end-length(filter)-1]

    ptimes = Int[-999]
    for i=1:length(trigger)
        if trigger[i] > 25 && i>10+ptimes[end]
            push!(ptimes, i)
        end
    end
    ptimes=ptimes[2:end]
    println("The average pulse rate is ",length(ptimes)/(ljhfile.dt*(ptimes[end]-ptimes[1])))

    use=9650:11230
    use=11810:14600
    use=4100:9800
    data = float(data[use]) - data[use[1]]
    trigger = trigger[use]
    const N = use[end]-use[1]+1
    ptimes = ptimes[ptimes .> use[1]]
    ptimes = ptimes[ptimes .< use[end]]
    ptimes = ptimes - use[1]

    clf()
    figure(5, figsize=(12, 7))
    subplots_adjust(top=.95, bottom=.05, left=.05, right=.95)
    const OFFSET_TRIG = 900
    const OFFSET_COMPS = -700
    const OFFSET_MODEL = -1300
    const OFFSET_DIFF = -1450
    const XTEXT = -70
    xlim([-450, N])
    ylim([OFFSET_DIFF-150, OFFSET_TRIG+200])
    xticks([])
    yticks([])

    plot(trigger+OFFSET_TRIG, "purple")
    text(XTEXT, OFFSET_TRIG,"Trig", va="center", ha="right", fontsize="large")
    plot(data, "b", lw=2)
    text(XTEXT,0,"Raw", va="center", ha="right", fontsize="large")
    for p in ptimes
        plot([p,p], [OFFSET_TRIG-60, data[p+10]+50], color="gray")
    end

    # Get info from hdf5 file
    hdf5name = hdf5_name_from_ljh_name(filename, "mpf")
    h5file = h5open(hdf5name, "r")
    avg_pulse = read(h5file["chan1/average_pulse"])
    avg_dpdt = finite_diff_deriv(avg_pulse)
    covar_model = ExactCovarianceSolver(
        read_complex(h5file["chan1/noise/model_amplitudes"]),
        read_complex(h5file["chan1/noise/model_bases"]), 10000)
    close(h5file)

    # color map hack! I don't know how to use a colormap like
    # set_cmap("winter")
    color(x) = [0, x, 1-.5*x]  # winter
    color(x) = [1, x, 0]  # autumn
    color(x) = [1, .7*x, 0]
#    color(x) = [1, x, 1-x] # spring

    avg_pulse = vcat(avg_pulse, zeros(Float64, N))
    avg_dpdt = vcat(avg_dpdt, zeros(Float64, N))
    const np = length(ptimes)
    M = Array(Float64, N, 1+2*np)
    MW = Array(Float64, N, 1+2*np)
    for (i,p) in enumerate(ptimes)
        model = zeros(Float64, N)
        model[p-3:end] = avg_pulse[1:N-p+4]
        M[:,i] = model
        MW[:,i] = CovarianceModels.whiten(covar_model, model)
        plot(model*.2+OFFSET_COMPS+(i+length(ptimes))*20, color=color(i*.5/np+.5))#, color="#9966ff")
    end
    text(XTEXT, OFFSET_COMPS+np*20,"Cmps", va="center", ha="right", fontsize="large")
    for (i,p) in enumerate(ptimes)
        model = zeros(Float64, N)
        model[p-3:end] = avg_dpdt[1:N-p+4]
        plot(model+OFFSET_COMPS+i*20, color=color(i*.5/np))#, color="#ff66ff
        M[:,i+np] = model
        MW[:,i+np] = CovarianceModels.whiten(covar_model, model)
    end
    M[:,end] = 1.0
    MW[:,end] = CovarianceModels.whiten(covar_model, ones(Float64, N))
    A = MW'*MW
    param = A \ MW' * CovarianceModels.whiten(covar_model, data)
    println(sqrt(diag(inv(A))))
    rho = inv(A)
    for i=1:size(rho)[1]
        s = sqrt(rho[i,i])
        rho [i,:] /= s
        rho [:,i] /= s
    end
    println( param)
    for (i,p) in enumerate(ptimes)
        if i==8
            text(p, -60, "8,9", color="b")
        elseif i==10
            text(p, -60, "10,11", color="b")
        elseif i in (9,11)
            ;
        else
            text(p, -60, string(i), color="b")
        end
    end
    final_model = M * param
    plot(final_model+OFFSET_MODEL, "k")
    text(XTEXT, OFFSET_MODEL, "Model", va="center", ha="right", fontsize="large")
    plot((data-final_model)*10+OFFSET_DIFF, color="g")
    text(XTEXT, OFFSET_DIFF, "Diff\nx10", va="center", ha="right", fontsize="large")
    println("Residual: ", norm(data-final_model))

    data, trigger, param, rho
end


# Try to do the entire Multi-Pulse Fitting analysis starting with a pulse file
# and a raw file. Assumes Noise_analysis(noisename) is already done
# Use all 3 whiteners
function MPF_analysis_driver(setname::String="C_auto", whitenum::Int=1)
    const PATH="/Volumes/Data2014/Data/Data_CDM/2011_09_12"
    const cnum = 1
    filename = @sprintf("%s/2011_09_12_%s_chan%d.ljh", PATH, setname, cnum)
    println("Working on $filename with technique $whitenum")
    MPF_analysis_3whiteners(filename, false, whitenum)
end

function MPF_analysis_3whiteners(filename::String, forceNew::Bool=false, whitenum::Int=1)
    file = MicrocalFiles.LJHFile(filename)
    hdf5name = hdf5_name_from_ljh_name(filename, "mpf")
    println("We are about to compute MPF results; store into\n$hdf5name")
    h5file = h5file_update(hdf5name)
    try
        const do_all_fits = true

        grpname=string("chan$(file.channum)")
        h5grp = g_create_or_open(h5file, grpname)
        trigger_times = read(h5grp["trigger_times"])
        const MAXSAMP = 400000  # 8000 for NSLS

        # Do the fits
        if forceNew || ! exists(h5grp, "mpf") | do_all_fits
            avg_pulse = read(h5grp["average_pulse"])
            dpdt = read(h5grp["average_dpdt"])

            # Try the model from lab book 6 page 124.
            if whitenum==1
                covar_model = ExactCovarianceSolver([82.5463,24.8891,13.6768,3.1515,0],
                                              [.0871940295,.9941810695,
                                               .9992481192,.9999235960], MAXSAMP)
                mpfgrp = g_create_or_open(h5grp, "mpf")
            elseif whitenum == 2
                covar_model = StreamCovarianceSolver([24.8891,13.6768,3.1515,82.5463],
                                              [.9941810695, .9992481192,.9999235960, 0.])
                mpfgrp = g_create_or_open(h5grp, "mpfstream")
            elseif whitenum==4
                covar_model = StreamCovarianceSolver([82.5463] , [0.])
                mpfgrp = g_create_or_open(h5grp, "mpfstreamX")
            else
                covar_model = BadCovarianceSolver([24.8891,13.6768,3.1515,82.5463],
                                              [.9941810695, .9992481192,.9999235960, 0.])
                mpfgrp = g_create_or_open(h5grp, "mpfbad")
            end

            println("Computing multi-pulse fits...\n")

            fitter = MultiPulseFitter(avg_pulse, dpdt, covar_model)
            extendMPF!(fitter, MAXSAMP)

            const DPDT = true
            ph, dpdts, dph, ddpdt, prior_level, residW, residR, baseline =
                multi_pulse_fit_file(file, trigger_times, fitter, DPDT)
            @show dph[1:10]

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


try
    MPF_analysis_driver("C_auto",4)
catch
    @show "There was an error"
finally
    @show "Hohoho"
end
