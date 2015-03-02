module CovarianceModels

export CovarianceModel,
ExactCovarianceSolver,
StreamCovarianceSolver,
noisecovariance,
choleskyproduct,
covarproduct,
choleskysolve,
whiten,
unwhiten

# Several experimental whiteners:
abstract CovarianceModel


function toeplitz(c1::Vector, r1::Vector)
    nr = length(c1)
    nc = length(r1)
    t = zeros(eltype(c1), nr, nc)
    for r=1:nr
        for c=1:r
            t[r,c] = c1[1+r-c]
        end
        for c=r+1:nc
            t[r,c] = r1[1+c-r]
        end
    end
    return t
end

function toeplitz(c1::Vector)
    toeplitz(c1, c1)
end


# Cholesky decomposition of covariance matrix R, where
# R_{ij} = r(|j-i|) = \sum_{m=1}^k a(m) b(m)^|j-i| is real for i,j = 1,2,...
# and |b(m)|<1 for m = 1,...,k.

FloatOrComplex = Union(FloatingPoint, Complex)

immutable ExactCovarianceSolver{T<:FloatOrComplex} <: CovarianceModel
    num_exps          ::Int64     # was: k
    max_length        ::Int64     # was: nsav
    bases             ::Vector{T}  # was: b
    amplitudes        ::Vector{T}  # was not previously saved
    d                 ::Matrix{T}
    dsum              ::Vector{Float64}
    dsuminv           ::Vector{Float64}

    # Constructor takes a vector of k amplitudes; k (or k-1) exponential
    # bases; and n, the maximum length of vector to be modeled. If only k-1 bases
    # are supplied, then the last (kth) base is assumed to be very small, so that
    # its powers are approximately [1,0,0,...].

    function ExactCovarianceSolver(a::Vector{T}, b::Vector{T}, n::Integer)
        if length(a)==length(b)+1
            const verysmall = minimum(abs(b)) *1e-5
            b = [b, verysmall];
        elseif length(a)!=length(b)
            error("ExactCovarianceSolver: a,b must be vectors of same length")
        end

        k=length(a)
        d=zeros(T,k,n)
        bb=zeros(T,k,k)
        accum=zeros(T,k,k)
        dsum=zeros(Float64,n)
        dsuminv=zeros(Float64,n)
        s=1/sqrt(real(sum(a)))
        for i=1:k
            d[i,1]=a[i]*s
        end
        for i=1:k, j=1:k
            bb[j,i]=conj(b[j])*b[i]
            accum[j,i]=conj(d[j,1])*d[i,1]*bb[j,i]
        end
        # Proceed row by row to factor
        v=zeros(T,k)
        for m=2:n
            for i=1:k
                v[i]=a[i]-sum(accum[:,i])
            end
            s=1/sqrt(real(sum(v)))
            for i=1:k
                d[i,m]=v[i]*s
            end
            for i=1:k, j=1:k
                accum[j,i]=(accum[j,i]+conj(d[j,m])*d[i,m])*bb[j,i]
            end
        end
        for m=1:n
            s=0.0
            for i=1:k
                s+=real(d[i,m])
            end
            dsum[m]=s
            dsuminv[m]=1/dsum[m]
        end
        new(k, n, copy(b), copy(a), d, dsum, dsuminv)
    end
end

ExactCovarianceSolver{T<:FloatOrComplex}(a::Vector{T}, b::Vector{T}, n::Integer) =
    ExactCovarianceSolver{T}(a, b, n)


# Generate the model noise covariance function out to arbitrary length

function noisecovariance(m::CovarianceModel, n::Integer)
    R = zeros(Float64, n)
    R[1] = sum(real(m.amplitudes))
    # Outer loop is over the exponentials i; inner loop is over lags j.
    for i = 1:m.nexp
        a,b = m.amplitudes[i], m.bases[i]
        term = a*b
        for j = 2:n
            R[j] += real(term)
            term *= b
            abs(term)<1e-12 && break
        end
    end
    R
end



# Matrix-vector product L x = y, where R = L L' is the covariance matrix
# described above. The number # of unknowns n can be less than or equal to
# the size for which the model was created.

function choleskyproduct{T<:Number}(model::ExactCovarianceSolver, x::Array{T,1},
                                 leadingzeros::Integer=0)
    const n=length(x)
    if n>model.max_length
        error("choleskyproduct: length(x) greater than that supported in the CovarianceModel")
    end
    covarproduct(model, x, leadingzeros, false)
end

function covarproduct{T<:Number}(model::ExactCovarianceSolver, x::Array{T,1},
                                 leadingzeros::Integer=0, both::Bool=true)
    const n=length(x)
    if n>model.max_length
        error("covarproduct: length(x) greater than that supported in the CovarianceModel")
    end
    const conjb=conj(model.bases)
    const k = model.num_exps
    const d = model.d
    const dsum = model.dsum
    InternalType = typeof(model.bases[1])
    y=zeros(T, n)
    if both
        #     ss=zeros(InternalType, k)
        error("Joe has not written a full covarproduct for Rx=y computation")
    end
    ss=zeros(InternalType, k)
    for i = 1:n
        y[i] = x[i]*dsum[i]+real(sum(ss))
        for j = 1:k
            ss[j] = (ss[j]+x[i]*conj(d[j,i]))*conjb[j]
        end
    end
    return y
end

# Solves with multiple right hand sides
function choleskyproduct{T<:Number}(model::ExactCovarianceSolver, x::Array{T,2})
    const n,m = size(x)
    if n>model.max_legnth
        error("choleskyproduct: length(y) greater than supported in the CovarianceModel")
    end
    const conjb=conj(model.bases)
    const k = model.num_exps
    const d = model.d
    const dsum = model.dsum
    const InternalType = typeof(model.bases[1])
    y=zeros(T,n,m)
    for j = 1:m
        ss=zeros(InternalType, k)
        for i=1:n
            y[i,j] = x[i,j]*dsum[i]+real(sum(ss))
            for l = 1:k
                ss[l]=(ss[l]+x[i,j]*conj(d[l,i]))*conjb[l]
            end
        end
    end
    return y
end

# Solve linear system L x = y, where R = L L' is the covariance matrix
# described above. The number of unknowns n can be less than or equal to
# the size for which the model was created.

function choleskysolve{T<:Number}(model::ExactCovarianceSolver, y::Array{T,1},
                              leadingzeros::Integer=0)
    const n=length(y)
    if n>model.max_length
        error("choleskysolve: length(y) greater than that supported in the CovarianceModel")
    end
    covarsolve(model, y, leadingzeros, false)
end

# Solve linear system R x = y, where R is the covariance matrix
# described above. The number of unknowns n can be less than or equal to
# the size for which the model was created. Since this shares much code
# with choleskysolve, that function calls this one
#
# If leadingzeros > 0 (0 is the default), then the function will effectively
# prepend that many leading zeros before y. Although these zeros do not appear
# in the INPUT, they DO APPEAR IN THE OUTPUT

function covarsolve{T<:Number}(model::ExactCovarianceSolver, y::Array{T,1},
                               leadingzeros::Integer=0, both::Bool=true)
    const n=length(y)+leadingzeros
    if n>model.max_length
        error("covarsolve: length(y) greater than that supported in the CovarianceModel")
    end
    x=zeros(T,n)
    const conjb=conj(model.bases)
    const k = model.num_exps
    const d = model.d
    const dsuminv = model.dsuminv
    const InternalType = eltype(model.bases)
    ss=zeros(InternalType, k)
    for i = 1+leadingzeros:n
        x[i] = (y[i-leadingzeros]-real(sum(ss)))*dsuminv[i]
        for j = 1:k
            ss[j] =(ss[j]+x[i]*conj(d[j,i]))*conjb[j]
        end
    end
    if both
        const b = model.bases
        ss=zeros(InternalType, k)
        for i = n:-1:1
            sds = 0.0
            for j = 1:k
                sds += real(ss[j]*d[j,i])
            end
            x[i] = (x[i]-sds)*dsuminv[i]
            for j = 1:k
                ss[j] =(ss[j]+x[i])*b[j]
            end
        end
    end
    return x
end

# Solves with multiple right hand sides
# Note that we don't have an implementation of covarsolve for a 2D array. Do it
# if needed.

function choleskysolve{T<:Number}(model::ExactCovarianceSolver, y::Array{T,2})
    const n,m = size(y)
    if n>model.max_length
        error("choleskysolve: length(y[:,1]) greater than that supported in the CovarianceModel")
    end
    const conjb=conj(model.bases)
    const k = model.num_exps
    const d = model.d
    const dsuminv = model.dsuminv
    const InternalType = eltype(model.bases)
    x=zeros(T,n,m)
    for j=1:m
        ss=zeros(InternalType, k)
        for i=1:n
            x[i,j]=(y[i,j]-real(sum(ss)))*dsuminv[i]
            for l = 1:k
                ss[l] =(ss[l]+x[i,j]*conj(d[l,i]))*conjb[l]
            end
        end
    end
    return x
end


function derive_polynomials_from_covar{T<:FloatOrComplex}(a::Vector{T}, b::Vector{T})
    # Given covariance as a sum over exponentials with amplitudes in a and bases in b,
    # derive and return (phi,mu), the two polynomials representing the rational function
    # expression for the ARMA process.
    @assert maximum(abs(b)) < 1
    @assert length(a)==length(b)
    @assert b[1] == 0

    # First the phi polynomial. That's easy: phi(z) = product_i=1:p of (1-z/r_i) for {r_i}
    # are the roots of the polynomial. Note that for each base b, there is a root 1/b.
    # Note that we're representing the polynomial as [1, phi1, phi2, ...phi_p] from zero
    # to higher powers of z, and NOT the way numpy represents it.
    p = length(b)-1
    phicplx = zeros(Complex128, 1+p)
    phicplx[1] = 1
    if p >= 0
        for r in b
            if r != 0; phicplx[2:end] -= phicplx[1:end-1] * r; end
        end
    end
    phi = float(abs(phicplx))
    @show phi

    # Now the mu polynomial. That's trickier, because it is the solution of a nonlinear
    # system.  Starting guess comes from factoring a large-ish chunk of the infinite matrix:
    N = 500
    philong = zeros(Float64, N)
    philong[1:p+1] = phi
    P = toeplitz(philong, 0*philong)
    t = [0:N-1]
    Covar = zeros(Float64, N)
    Covar[1] = a[1]
    for i=2:p+1
        Covar += a[i] * b[i].^t
    end
    CovarMA = P*toeplitz(Covar)*P'
    B1 = float(chol(CovarMA, :L))
    mu_sig2 = vec([B1][end, end:-1:N-p])
    mu = mu_sig2/mu_sig2[1]
    @show mu
    # So far, that's only a starting guess to use in a nonlinear solver. But for now, let's
    # just use it. (If using a solver, you can reduce N to speed that part up.)

    (phi,mu)
end
#@show derive_polynomials_from_covar([82.5463,24.8891,13.6768, 3.1515],[0,.994180695,.9992481192,.9999235960])



type StreamCovarianceSolver{T<:FloatOrComplex} <: CovarianceModel
    bases             ::Vector{T}
    amplitudes        ::Vector{T}
    phi               ::Vector{T}
    theta             ::Vector{T}
    isinitialized     ::Bool
    inhistory         ::Vector{T}
    outhistory        ::Vector{T}


    # Constructor takes a vector of k amplitudes; k (or k-1) exponential
    # bases; and n, the maximum length of vector to be modeled. If only k-1 bases
    # are supplied, then the last (kth) base is assumed to be very small, so that
    # its powers are approximately [1,0,0,...].

    function StreamCovarianceSolver(a::Vector{T}, b::Vector{T})
        if length(a)==length(b)+1
            const verysmall = minimum(abs(b)) *1e-5
            b = [b, verysmall];
        elseif length(a)!=length(b)
            error("StreamCovarianceSolver: a,b must be vectors of same length")
        end

        (phi,theta) = derive_polynomials_from_covar(a, b)

        new(copy(b), copy(a), phi, theta, false, 0*phi, 0*theta)
    end
end

StreamCovarianceSolver{T<:FloatOrComplex}(a::Vector{T}, b::Vector{T}) =
    StreamCovarianceSolver{T}(a, b)


function _initialize_stream{T<:Number}(model::StreamCovarianceSolver, x::Array{T,1})
    const n=length(x)
    const p=length(model.phi)-1
    const q=length(model.theta)-1
    @assert p==q # for now
    if n<=p
        error("choleskysolve: length(y) too small to set up solver (implement this later")
    end
    initialPhi = toeplitz(model.phi, 0*model.phi)
    initialTheta = toeplitz(model.theta, 0*model.theta)
    x = x[1:p+1]
    out = initialTheta \  initialPhi * x
    model.inhistory[:] = x[:]
    model.outhistory[:] = out[:]
end


# Solve linear system L w = x, where R = L L' is the covariance matrix
# described above. The number of unknowns n can be less than or equal to
# the size for which the model was created.

function choleskysolve{T<:Number}(model::StreamCovarianceSolver, x::Array{T,1})
    const n=length(x)
    const p=length(model.phi)-1
    const q=length(model.theta)-1
    @assert p==q # for now

    w = copy(x)

    if !model.isinitialized
        _initialize_stream(model, x[1:p+1])
        w[1:p+1] = model.outhistory[:]
    else
        for i=1:p+1
            for j=2:i
                w[i] -= model.theta[j]*w[i+1-j]
                w[i] += model.phi[j]*x[i+1-j]
            end
            for j=i+1:q+1
                w[i] -= model.theta[j]*model.outhistory[q+i+1-j]
                w[i] += model.phi[j]*model.inhistory[q+i+1-j]
            end
        end
    end
    for i=p+2:n
        for j=2:q+1
            w[i] -= model.theta[j]*w[i+1-j]
            w[i] += model.phi[j]*x[i+1-j]
        end
    end
    model.inhistory = x[end-p:end]
    model.outhistory = w[end-q:end]
    return w
end



whiten = choleskysolve
unwhiten = choleskyproduct


end # module

#using PyPlot
#solver=StreamCovarianceSolver([82.5463,24.8891,13.6768, 3.1515],[0,.994180694,.9992481192,.9999235960]); clf(); plot( choleskysolve(solver, ones(Float64,1269)),"-b")


function main()
    a=[10.0+0.0im;
       0.194487+0.405512im;
       0.194487-0.405512im;
       -0.4358-0.0374477im;
       -0.4358+0.0374477im;
       0.4986+0.31128im;
       0.4986-0.31128im;
       0.385488-0.00129318im;
       0.385488+0.00129318im;
       -0.283494-0.291219im;
       -0.283494+0.291219im]
    b=[0.12+0.0im;
       -0.320372+0.797491im;
       -0.320372-0.797491im;
       0.720776+0.102379im;
       0.720776-0.102379im;
       0.370054-0.0357288im;
       0.370054+0.0357288im;
       -0.652465+0.506429im;
       -0.652465-0.506429im;
       0.696761+0.622623im;
       0.696761-0.622623im]

    reala = real(a)
    realb = [.12, -.3204, -.3942, .720, .820, .370, .320, -.6524, -.5490, .697, .650]
    n=1000000 # for 1000000 expect 5.5 sec, .08, .08 for 3 parts
    x=rand(n)

    print("Doing preparatory work   ")
    tic()
    #model = CovarianceModel(a,b,n)
    model = CovarianceModel(reala,realb,n)
    toc()
    print("Doing choleskyproduct    ")
    tic()
    y=choleskyproduct(model, x)
    toc()
    print("Doing choleskysolve      ")
    tic()
    z=choleskysolve(model, y)
    toc()
    print("Doing covarsolve         ")
    tic()
    u = covarsolve(model, x)
    toc()
    #print("Doing covarproduct       ")
    #tic()
    #v = covarproduct(model, u)
    #toc()
    println(norm(x-z)/norm(z))
    #println(norm(v-x)/norm(x))

end

# main()

    #y=covprodut(x, model.num_exps, model.max_length, model.bases, model.d, model.dsum, model.dsuminv)
    #z=covsolut(y, model.num_exps, model.max_length, model.bases, model.d, model.dsum, model.dsuminv)
# function covprodut{T<:Number}(x::Array{T,1},k,nsav,b,d,dsum,dsuminv)
#     const n=length(x)
#     if n>nsav
#         error("covprodut: length(x) greater than that provided in cholsav")
#     end
#     const conjb=conj(b)
#     ss=zeros(Complex128,k)
#     y=zeros(T,n)
#     for i=1:n
#         y[i]=x[i]*dsum[i]+real(sum(ss))
#         for j = 1:k
#             ss[j]=(ss[j]+x[i]*conj(d[j,i]))*conjb[j]
#         end
#     end
#     return y
# end


# function covsolut{T<:Number}(y::Array{T,1},k,nsav,b,d,dsum,dsuminv)
#     n=length(y)
#     if n>nsav
#         error("covsolut: length(y) greater than that provided in cholsav")
#     end
#     conjb=conj(b)
#     x=zeros(T,n)
#     ss=zeros(Complex128,k)
#     for i=1:n
#         x[i]=(y[i]-real(sum(ss)))*dsuminv[i]
#         #ss=(ss+x[i]*conj(d[:,i])).*conjb
#         for j = 1:k
#             ss[j] =(ss[j]+x[i]*conj(d[j,i]))*conjb[j]
#         end
#     end
#     return x
# end



