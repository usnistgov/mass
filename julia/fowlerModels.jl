module fowlerModels

export ARMAModel,
generateARMA,
PulseModel,
generatePulse



FloatOrComplex = Union(Real, Complex)


type ARMAModel{T<:FloatOrComplex}
    phi          ::Vector{T}
    theta        ::Vector{T}
    p            ::Int
    q            ::Int
    epshistory   ::Vector{Float64}
    outhistory   ::Vector{T}

    function ARMAModel(ar::Vector{T}, ma::Vector{T})
        p = length(ar)-1
        q = length(ma)-1
        scale = ar[1]
        ar = ar/scale
        ma = ma*scale
        new(ar, ma, p, q, zeros(Float64, q), zeros(Float64, p))
    end
end

ARMAModel{T<:FloatOrComplex}(ar::Vector{T}, ma::Vector{T}) =
    ARMAModel{T}(ar, ma)

function generateARMA(model::ARMAModel, n::Int)
    eps = Array(Float64, n)
    randn!(eps)
    eps = vcat(model.epshistory, eps)
    model.epshistory = eps[end-model.q+1:end]

    # Perform the MA part and store (temporarily) in out
    out = zeros(eltype(model.outhistory), n)
#    theta = model.theta[end:-1:1]
    for i=1:n
        for j=1:model.q+1
            out[i] += model.theta[j]*eps[i+model.q-j+1]
        end
    end
    out = vcat(model.outhistory, out)
    @assert model.phi[1] == 1.0
    for i=1+model.p:n+model.p
        for j=1:model.p
            out[i] -= model.phi[j+1]*out[i-j]
        end
    end
    model.outhistory = out[end-model.p+1:end]
    return real(out[1+model.p:end])
end


type PulseModel{T<:FloatOrComplex}
    decaybase     ::Vector{T}
    pulseampl     ::Vector{T}
    prevampl      ::Vector{T}

    function PulseModel(bases::Vector{T}, ampl::Vector{T})
        @assert length(bases) == length(ampl)
        new(bases, ampl, zeros(T, length(ampl)))
    end
end

PulseModel{T<:Real}(b::Vector{T}, a::Vector{T})  = PulseModel{T}(float(b), float(a))
PulseModel{T<:Complex}(b::Vector{T}, a::Vector{T}) = PulseModel{T}(complex(b), complex(a))

function generatePulse(model::PulseModel, n::Int, arrivaltimes::Vector{Int})
    out = Array(Float64, n)
    nextat = -1
    if length(arrivaltimes) > 0
        nextat = arrivaltimes[1]
        arrivaltimes = arrivaltimes[2:end]
    end
    for i=1:n
        if i==nextat
            if length(arrivaltimes) > 0
                nextat = arrivaltimes[1]
                arrivaltimes = arrivaltimes[2:end]
            else
                nextat=-1
            end
            model.prevampl += model.pulseampl
        end
        out[i] = sum(model.prevampl)
        model.prevampl = model.prevampl .* model.decaybase
    end
    return out
end


end # module