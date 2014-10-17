

using HDF5, PyPlot
using MicrocalFiles, HDF5Helpers

include("NoiseAnalysis.jl")

# Try on "/Volumes/Data2014/Data/NSLS_data/2012_06_14/2012_06_14_S_chan93.ljh"
# Noise  "/Volumes/Data2014/Data/NSLS_data/2012_06_14/2012_06_14_V_chan93.noi"

# type MultiPulseFits
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
        @show forceNew
        @show exists(h5grp, "trigger_times")
        if forceNew || ! exists(h5grp, "trigger_times")
            trig_times = find_all_pulses(file)
            ds_update(h5grp, "trigger_times", trig_times)
        end
        ttds = h5grp["trigger_times"]

        # Analyze the noise
        if forceNew || ! exists(h5grp, "noise")
            data = MicrocalFiles.fileData(noisename)
            deblip_nsls_data!(data, -10)
            nlags, nexps = 2048, 4
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

    finally
        close(h5file)
    end
    0
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