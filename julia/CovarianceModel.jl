
# Cholesky decomposition of covariance matrix R, where
# R_{ij} = r(|j-i|) = \sum_{m=1}^k a(m) b(m)^|j-i| is real for i,j = 1,2,...
# and |b(m)|<1 for m = 1,...,k.

immutable CovarianceModel
    num_exps          ::Int64     # was: k
    max_length        ::Int64     # was: nsav
    bases             ::Vector{Complex128}  # was: b
    amplitudes        ::Vector{Complex128}  # was not previously saved
    d                 ::Matrix{Complex128}
    dsum              ::Vector{Float64}
    dsuminv           ::Vector{Float64}

    # Constructor takes a vector of k amplitudes; k (or k-1) exponential
    # bases; and n, the maximum length of vector to be modeled. If only k-1 bases
    # are supplied, then the last (kth) base is assumed to be very small, so that
    # its powers are approximately [1,0,0,...].

    function CovarianceModel(a::Vector{Complex128}, b::Vector{Complex128}, n::Integer)
        if length(a)==length(b)+1
            const verysmall = minimum(abs(b)) *1e-5
            b = [b, verysmall];
        elseif length(a)!=length(b)
            error("CovarianceModel: a,b must be vectors of same length")
        end
        
        k=length(a)
        d=zeros(Complex128,k,n)
        bb=zeros(Complex128,k,k)
        accum=zeros(Complex128,k,k)
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
        v=zeros(Complex128,k)
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

# Alternate constructor for non-complex bases/amplitudes
function CovarianceModel{T<:FloatingPoint}(a::Vector{T}, b::Vector{T}, n::Integer)
    CovarianceModel(complex(a), complex(b), n)
end


# Generate the model noise covariance function out to arbitrary length

function noisecovariance(m::CovarianceModel, n::Integer)
    R = zeros(Float64, n)
    R[1] = sum(real(m.amplitudes))
    # Outer loop is over the exponentials i; inner loop is over lags j.
    for i = 1:m.nexp
        a,b = m.amplitudes[i], m.bases[i]
        powerb = b
        for j = 2:n
            R[j] += real(a*powerb)
            powerb *= b
            abs(powerb)<1e-12 && break
        end
    end
    R
end



# Matrix-vector product U' x = y, where R = U' U is the covariance matrix
# described above and cholsav was produced by covchol above. The number
# of unknowns n can be less than or equal to the size for which covchol
# was created.

function covprodut{T<:Number}(x::Array{T,1},k,nsav,b,d,dsum,dsuminv)
    const n=length(x)
    if n>nsav
        error("covprodut: length(x) greater than that provided in cholsav")
    end
    const conjb=conj(b)
    ss=zeros(Complex128,k)
    y=zeros(T,n)
    for i=1:n
        y[i]=x[i]*dsum[i]+real(sum(ss))
        for j = 1:k
            ss[j]=(ss[j]+x[i]*conj(d[j,i]))*conjb[j]
        end
    end
    return y
end

function covarproduct{T<:Number}(model::CovarianceModel, x::Array{T,1}, 
                                 leadingzeros::Integer=0)
    const n=length(x)
    if n>model.max_length
        error("covarproduct: length(x) greater than that supported in the CovarianceModel")
    end
    const conjb=conj(model.bases)
    const k = model.num_exps
    const d = model.d
    const dsum = model.dsum
    ss=zeros(Complex128, k)
    y=zeros(T, n)
    for i = 1:n
        y[i] = x[i]*dsum[i]+real(sum(ss))
        for j = 1:k
            ss[j] = (ss[j]+x[i]*conj(d[j,i]))*conjb[j]
        end
    end
    return y
end
# Solves with multiple right hand sides
function covarproduct{T<:Number}(model::CovarianceModel, x::Array{T,2})
    const n,m = size(x)
    if n>model.max_legnth
        error("covarproduct: length(y) greater than supported in the CovarianceModel")
    end
    const conjb=conj(model.bases)
    const k = model.num_exps
    const d = model.d
    const dsum = model.dsum
    y=zeros(T,n,m)
    for j = 1:m
        ss=zeros(Complex128,model.num_exps)
        for i=1:n
            y[i,j] = x[i,j]*dsum[i]+real(sum(ss))
            for l = 1:k
                ss[l]=(ss[l]+x[i,j]*conj(d[l,i]))*conjb[l]
            end
        end
    end
    return y
end

# Solve linear system U' x = y, where R = U' U is the covariance matrix
# described above and cholsav was produced by covchol above. The number
# of unknowns n can be less than or equal to the size for which covchol
# was created.

function covsolut{T<:Number}(y::Array{T,1},k,nsav,b,d,dsum,dsuminv)
    n=length(y)
    if n>nsav
        error("covsolut: length(y) greater than that provided in cholsav")
    end
    conjb=conj(b)
    x=zeros(T,n)
    ss=zeros(Complex128,k)
    for i=1:n
        x[i]=(y[i]-real(sum(ss)))*dsuminv[i]
        #ss=(ss+x[i]*conj(d[:,i])).*conjb
        for j = 1:k
            ss[j] =(ss[j]+x[i]*conj(d[j,i]))*conjb[j]
        end
    end
    return x
end

function covarsolve{T<:Number}(y::Array{T,1}, model::CovarianceModel,
                              leadingzeros::Integer=0)
    const n=length(y)
    if n>model.max_length
        error("covarsolve: length(y) greater than that supported in the CovarianceModel")
    end
    x=zeros(T,n)
    const conjb=conj(model.bases)
    const k = model.num_exps
    const d = model.d
    const dsuminv = model.dsuminv
    ss=zeros(Complex128, k)
    for i = 1+leadingzeros:n
        x[i] = (y[i]-real(sum(ss)))*dsuminv[i]
        for j = 1:k
            ss[j] =(ss[j]+x[i]*conj(d[j,i]))*conjb[j]
        end
    end
    return x
end
# Solves with multiple right hand sides
function covarsolve{T<:Number}(y::Array{T,2}, model::CovarianceModel)
    const n,m = size(y)
    if n>model.max_length
        error("covarsolve: length(y[:,1]) greater than that supported in the CovarianceModel")
    end
    const conjb=conj(model.bases)
    const k = model.num_exps
    const d = model.d
    const dsuminv = model.dsuminv
    x=zeros(T,n,m)
    for j=1:m
        ss=zeros(Complex128, k)
        for i=1:n
            x[i,j]=(y[i,j]-real(sum(ss)))*dsuminv[i]
            for l = 1:k
                ss[l] =(ss[l]+x[i,j]*conj(d[l,i]))*conjb[l]
            end
        end
    end
    return x
end

whiten = covarsolve


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
    n=1000000 # for 1000000 expect 5.5 sec, .08, .08 for 3 parts
    x=rand(n)

    tic()
    model = CovarianceModel(a,b,n)
    toc()
    tic()
    y=covarproduct(model, x)
    #y=covprodut(x, model.num_exps, model.max_length, model.bases, model.d, model.dsum, model.dsuminv)
    toc()
    tic()
    z=covarsolve(model, y)
    #z=covsolut(y, model.num_exps, model.max_length, model.bases, model.d, model.dsum, model.dsuminv)
    toc()
    println(norm(x-z)/norm(z))
end

#main()
