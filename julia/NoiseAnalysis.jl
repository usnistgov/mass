# Functions to analyze the noise data ??
# Joe Fowler, NIST
# October, 2014

using MicrocalFiles
using HDF5, HDF5Helpers
using PyPlot

type NoiseSummary
    noise_autocorr    ::Vector{Float64}
    nlags             ::Integer

    nexps             ::Integer
    exp_bases         ::Vector{Complex128}
    exp_amplitudes    ::Vector{Complex128}
    model_autocorr    ::Vector{Float64}
    model_rms         ::Float64


    function NoiseSummary(nlags::Integer, nexps::Integer)
        noise_autocorr = Array(Float64, nlags)
        model_autocorr = Array(Float64, nlags)
        exp_bases = Array(Complex128, nexps)
        exp_amplitudes = Array(Complex128, nexps+1)
        new(noise_autocorr, nlags, nexps, exp_bases, exp_amplitudes,
            model_autocorr, 0.0)
    end
end


# Generate the HDF5 summary for an LJH file by filename
summarize_noise(filename::String, nlags::Integer, nexps=4) =
    summarize_noise(MicrocalFiles.LJHFile(filename), nlags, nexps)



# Summarize and model the noise in a noise file object, and store it into
# an HDF5 file

function summarize_noise(file::MicrocalFiles.LJHFile, nlags::Integer, nexps=4)
    hdf5name = hdf5_name_from_ljh_name(file.name)
    println("We are about to summarize noise data into '$hdf5name'")
    if isreadable(hdf5name)
        h5file = h5open(hdf5name, "r+")
    else
        h5file = h5open(hdf5name, "w")
    end
    grpname=string("chan$(file.channum)")
    h5grp = g_create_or_open(h5file, grpname)

    summary = compute_noise_summary(file.name, nlags, nexps)

    summgrp = g_create_or_open(h5grp,"noise")
    # Store basic information
#     attr_update(h5grp, "npulses", summary.nrec)
#     attr_update(h5grp, "nsamples", summary.nsamp)
#     attr_update(h5grp, "npresamples", summary.npre)
#     attr_update(h5grp, "frametime", summary.dt)
    attr_update(summgrp, "rawname", file.name)
    attr_update(summgrp, "nlags", summary.nlags)
    attr_update(summgrp, "nexps", summary.nexps)
    attr_update(summgrp, "model_rms", summary.model_rms)
    ds_update(summgrp, "autocorrelation", summary.noise_autocorr)
    ds_update(summgrp, "model_autocorr", summary.model_autocorr)
    ds_update(summgrp, "model_amplitudes", summary.exp_amplitudes)
    ds_update(summgrp, "model_bases", summary.exp_bases)
    close(h5file)
    summary
end


# Summarize and model the noise in a noise file. Do not save it to
# any kind of external file.

function compute_noise_summary(filename::String, nlags::Integer, nexps=4)
    data = MicrocalFiles.fileData(filename)
    compute_summary(data, nlags, nexps)
end


function compute_noise_summary(data::Vector, nlags::Integer, nexps=4)
    summary = NoiseSummary(nlags, nexps)
    summary.noise_autocorr = autocorrelation(data, nlags)

    as, bs = fit_exponential_model(summary.noise_autocorr, nexps)
    summary.exp_amplitudes = as
    summary.exp_bases = bs

    model = zeros(Float64, nlags)
    model[1] = sum(real(as[1:end]))
    const j = 1:nlags-1
    for i in 1:nexps
        b = bs[i]
        model[2:end] += real(as[i] * b.^j)
    end
    summary.model_autocorr = model
    summary.model_rms = sqrt(mean((model-summary.noise_autocorr).^2))
    summary
end


function plot_noise_summary(s::NoiseSummary)
    clf()
    plot(s.noise_autocorr,"r", label="data")
    plot(s.model_autocorr,"b", label="model")
    plot(s.noise_autocorr-s.model_autocorr,"g",label="residual")
    legend(loc="upper right")
end


# From a vector of data, generate the autocorrelation function for
# lags 0:nlags-1. Do this using sequences of length seqlen >= nlags*2
# After 0-padding the sequences, an FFT will be performed on data of
# length (seqlen+nlags), so suggest this number be a power of 2 or at
# least have a small # of small prime factors.

# The optional argument max_excursion will cut a sequence if any sample
# in it differs from the mean by more than max_excursion.

function autocorrelation(data::Vector, nlags::Integer, seqlen::Integer,
    max_excursion=200.0)

    println("lags: $nlags, seqlen: $seqlen")
    @assert seqlen >= 2*nlags
    @assert seqlen < length(data)

    nseq = div( length(data), seqlen)
    datamean = mean(data)
    d = zeros(Float64, seqlen+nlags)
    acorr = zeros(Float64, nlags)
    segments_used = 0

    for i = 1:nseq
        datamean = mean(data[(i-1)*seqlen+1:i*seqlen])
        d[1:seqlen] = float(data[(i-1)*seqlen+1:i*seqlen]) - datamean
        if maximum(d) > max_excursion || minimum(d) < -max_excursion
            continue
        end
        acorr += xcorr(d, d)[seqlen+nlags:seqlen+2*nlags-1]
        segments_used += 1
    end
    if segments_used < nseq
        println("Used $segments_used segments out of $nseq available.")
    end
    acorr ./ (segments_used*[seqlen:-1:seqlen-nlags+1])
end


function autocorr_seqlen_heuristic(nlags::Integer)
    # We'll use sequences at least 7x the # of lags, then
    # round up to one of (1,3,5) times a power of 2
    total =8*nlags
    if total < 8
        total = 8
    end

    lowpow2 = 2^int(floor(log2(total)))
    pow2ratio = total/lowpow2
    if pow2ratio > 1.5
        total = lowpow2*2
    elseif pow2ratio > 1.25
        total = div(lowpow2*3, 2)
    else
        total = div(lowpow2*5, 4)
    end
    @assert total >= 8*nlags
    total-nlags
end


# Another method, where the sequence length is chosen by a simple
# heuristic: 7x the # of lags, or more if needed to reach a size
# that can be readily FFTed.

autocorrelation(data::Vector, nlags::Integer) =
    autocorrelation(data, nlags, autocorr_seqlen_heuristic(nlags))


include("RandomMatrix.jl")


# Find the eigenvalues of the "system matrix", i.e the matrix that
# connects data[2:end] and data[1:end-1].  The technique comes from
# "The fit of a sum of exponentials to noisy data" by P de Groen and B de
# Moor, J. Computational and Applied Math 20, 175-187 (1987).
# Will produce nval eigenvalues using the nval leading components of
# a certain SVD.  (A plot of the leading singular values is made using
# 3*nval values.)

function estimate_exponentials(data::Vector, nval::Integer)
    N = length(data)

    # Create the Hankel matrix of data values
    width = div(N, 5)
    height = N-width+1
    println("Size of H: $height*$width")
    @assert width*height < 50000000
    H = Array(typeof(data[1]), height, width)
    for c in 1:width
        for r in 1:height
            H[r, c] = data[c+r-1]
        end
    end

    # Find the SVD of the Hankel matrix, computing 3 times as many
    # values as we'll ultimately use
    u,w,v = find_svd_randomly(H[1:end-1,1:end], 3*nval)
    println("Singular values: ",w)
    clf()
    semilogy(w,"o-r")

    # Estimate M, a matrix _similar to_ the system matrix A. This is Approach 1
    # due to Zeiger & McEwan (IEEE Tran Aut Control AC-19, 1953), as given in
    # de Groen and de Moor
    Hbar = H[2:end,1:end]
    Sigma_inv12 = diagm(w[1:nval] .^ (-.5))
    Pplus = Sigma_inv12 * u[1:end, 1:nval]'
    QPlus = v[1:end, 1:nval] * Sigma_inv12
    M = Pplus * Hbar * QPlus

    # By similarity, M has the same eigenvalues as A.
    evals = eigvals(M)
    evals#*(1.0+0im)   # Make sure we return complex
end



# Given a data array and a pre-computed array of exponential bases,
# compute the amplitude of each base in a least-squares fit.
# The data[1] (lag 0) value is left out of the fit, and the amplitude
# of the delta function required to fit data[1] exactly is returned
# as the last value of the array.
#
# Thus the returned array will be 1 longer than length(bases).

function fit_exponential_amplitudes_plusdelta(data::Vector{Float64},
                                              bases::Vector)
    N = length(data)
    M = length(bases)
    const j = 1:N-1
    model = [b^p for p=j, b in bases]
    amplitudes = pinv(model) * data[2:end]

    # Plot results
    fit = vcat(data[1], real(model*amplitudes))
    r = sqrt(mean((data-fit).^2))
    println ("Residual rms difference between model and fit $r")
    all_amplitudes = vcat(amplitudes, data[1]-real(sum(amplitudes)))
end



function fit_exponential_model(autocorr::Vector, nval::Integer)
    bases = estimate_exponentials(autocorr[2:end], nval)
    amplitudes = fit_exponential_amplitudes_plusdelta(autocorr, bases)
    amplitudes, bases
end


function build_phi(bases::Vector)
    phi = zeros(typeof(bases[1]), 1+length(bases))
    phi[1] = 1
    for b in bases
        phi[2:end] -= b*phi[1:end-1]
    end
    real(phi)
end



type NoiseModel
    exp_bases            ::Vector{Complex128}
    exp_amplitudes       ::Vector{Complex128}
    phi                  ::Vector{Float64}
    max_white_length     ::Integer
    nexp                 ::Integer
    Mcorner              ::Array{Float64,2}
    Bcorner              ::Array{Float64,2}
    Bbands               ::Array{Float64,2}

    function NoiseModel{T<:Number,U<:Number}(bases::Vector{T}, amplitudes::Vector{U},
                        maxlen::Integer)
        NoiseModel(complex128(bases), complex128(amplitudes), maxlen)
    end

    function NoiseModel(bases::Vector{Complex128}, amplitudes::Vector{Complex128},
                        maxlen::Integer)
        nexp = length(bases)
        @assert length(amplitudes) == 1+nexp

        phi = build_phi(bases)
        Mcorner = zeros(Float64, nexp+1, nexp+1)
        Bcorner = zeros(Float64, nexp+1, nexp+1)
        Bbands = zeros(Float64, maxlen, nexp+1)

        model = new(bases, amplitudes, phi, maxlen, nexp, Mcorner,
             Bcorner, Bbands)

        m = covar2MAcovar(model)
        # Shift the Bcorner matrix over so that diaonals(-p...0) are in col [1:p+1]
        for i=1:nexp+1
            model.Bbands[i,end-i+1:end] = model.Bcorner[i,1:i]
        end
        # Now compute the rest of Bbands
        for i=nexp+2:maxlen
            model.Bbands[i, 1] = m[1] / model.Bbands[i-nexp, end]
            for q=nexp-1:-1:1
                s = sum(model.Bbands[i, 1:end-q-1]  .* model.Bbands[i-q, q+1:end-1])
                model.Bbands[i, end-q] = (m[end-q] - s) / model.Bbands[i-q, end]
            end
            s = sum(model.Bbands[i,1:end-1].^2)
            model.Bbands[i, end] = sqrt(m[end] - s)
        end
        model
    end
end


function noise_covariance(m::NoiseModel, n::Integer)
    R = zeros(Float64, n)
    R[1] = sum(real(m.exp_amplitudes))
    # Outer loop is over the exponentials i; inner loop is over lags j.
    for i = 1:m.nexp
        a,b = m.exp_amplitudes[i], m.exp_bases[i]
        bi = b
        for j = 2:n
            R[j] += real(a*bi)
            bi *= b
            abs(bi)<1e-12 && break
        end
    end
    R
end


# We can think of the full model's covariance as R where M=ARA' and
# M=BB'. In this function, we compute M because the model gives us R
# and the A matrix.
# M will be computed to size (p+1)x(p+1), where p is the number of exponentials.
# In general, the last row (or column) of this small M dictactes all additional
# rows of a larger M, because M itself is Toeplitz except for the pxp upper left corner.

function covar2MAcovar(m::NoiseModel)
    p = m.nexp
    r = noise_covariance(m, p+1)
    R = toeplitz(r)
    A = toeplitz(m.phi, zeros(Float64, p+1))
    m.Mcorner = A*R*A'
    m.Bcorner = chol(m.Mcorner, :L)
    m.Mcorner[end,:]
end


# Whiten a data stream v using the causal whitening filter implied by
# the m::NoiseModel and return the whitened result.
function whiten{T<:Number}(m::NoiseModel, v::Vector{T})
    w = Array(Float64, length(v))
    whiten!(m, v, w)
    w
end



# Whiten a data stream v using the causal whitening filter implied by
# the m::NoiseModel. The result is returned in w

function whiten!{T<:Number}(m::NoiseModel, v::Vector{T}, w::Vector{Float64},
                 first_nonzero::Integer=1)
    N = length(v)
    p = m.nexp
    @assert length(w) >= N

    # Note that is(v, float(v)) will be true when v is already Vector{Float64}.
    v = float(v)

    # A shortcut will involve ignoring all leading zeros in v. Leading zeros
    # in v produce leading zeros in w, because whitening is linear and causal.
    # If first_nonzero is passed in from outside, we'll trust it. Otherwise
    # compute it.
    if first_nonzero <= 1
        first_nonzero = N
        for i in 1:N
            if v[i] != 0.0
                first_nonzero = i
                break
            end
        end
    end
    num_nonzero = N+1-first_nonzero

    # 1) Apply the A matrix. Result w is "half-whitened" in that the
    # AR part of the model is removed, but MA part still to be done.
    w[1:first_nonzero-1]  = 0.0
    w[first_nonzero:N] = conv(v[first_nonzero:N], m.phi)[1:num_nonzero]

    # 2) Solve Bw=a=Av.
    # 2a) First the small case where direct inversion makes sense.
    if N <= p+1
        w[1:N] = m.Bcorner[1:N,1:N] \ w[1:N]
    end

    # 2b) Now the larger Bw=a solution. Note that we have computed the rows of banded
    # B only to some limit. Beyond that limit, we can continue using the last row
    # of B and hope that B has converged well enough for this to be safe.
    Nb = m.max_white_length # This is # of rows in m.Bcorner
    if first_nonzero <= p+1
        w[1:p+1] = m.Bcorner \ w[1:p+1]
    end
    start_at = max(p+2, first_nonzero)
    for i = start_at:min(Nb, N)
        for j=i-p:i-1
            w[i] -= m.Bbands[i, j-i+p+1] * w[j]
        end  # s = sum(m.Bbands[i, 1:end-1] * w[i-p:i-1])
        w[i] /= m.Bbands[i,end]
    end
    # From here on out, we have to approximate un-computed rows of B by the
    # last computed row and hope it works.
    start_at = max(Nb+1, first_nonzero)
    for i = start_at:N
        for j=i-p:i-1
            w[i] -= m.Bbands[end, j-i+p+1] * w[j]
        end  # s = sum(m.Bbands[end, 1:end-1] * w[i-p:i-1])
        w[i] /= m.Bbands[end,end]
    end
end



# Create and return a toeplitz matrix with c as the first column and
# (optional) r as the first row. r[1] will be ignored in
# favor of c[1] to determine the diagonal.

function toeplitz{T<:Number}(c::Vector{T}, r::Vector{T})
    nr = length(c)
    nc = length(r)
    m = Array(typeof(c[1]), nr, nc)
    for j=1:nc
        for i=1:j-1  # Strict upper triangle
            m[i,j] = r[j-i+1]
        end
        for i=j:nr
            m[i,j] = c[i-j+1]
        end
    end
    m
end


# Create and return a symmetric toeplitz matrix given the first column c

toeplitz{T<:Number}(c::Vector{T}) = toeplitz(c,c)
