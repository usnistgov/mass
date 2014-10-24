# Cholesky decomposition of covariance matrix R, where
# R_{ij} = r(|j-i|) = \sum_{m=1}^k a(m) b(m)^|j-i| is real for i,j = 1,2,...
# and |b(m)|<1 for m = 1,...,k.

function covchol(a,b,n)
    if length(a)!=length(b)
        error("covchol: a,b must be vectors of same length")
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
    cholsav = k,n,copy(b),d,dsum,dsuminv
    return cholsav
end


# Matrix-vector product U' x = y, where R = U' U is the covariance matrix
# described above and cholsav was produced by covchol above. The number
# of unknowns n can be less than or equal to the size for which covchol
# was created.

function covprodut{T<:Number}(x::Array{T,1},k,nsav,b,d,dsum,dsuminv)
    n=length(x)
    if n>nsav
        error("covprodut: length(x) greater than that provided in cholsav")
    end
    conjb=conj(b)
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
# Solves with multiple right hand sides
function covprodut{T<:Number}(x::Array{T,2},k,nsav,b,d,dsum,dsuminv)
    n,m = size(x)
    if n>nsav
        error("covsolu: length(y) greater than that provided in cholsav")
    end
    conjb=conj(b)
    y=zeros(T,n,m)
    for j=1:m
        ss=zeros(Complex128,k)
        for i=1:n
            y[i,j]=x[i,j]*dsum[i]+real(sum(ss))
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
# Solves with multiple right hand sides
function covsolut{T<:Number}(y::Array{T,2},k,nsav,b,d,dsum,dsuminv)
    n,m = size(y)
    if n>nsav
        error("covsolut: length(y[:,1]) greater than that provided in cholsav")
    end
    conjb=conj(b)
    x=zeros(T,n,m)
    for j=1:m
        ss=zeros(Complex128,k)
        for i=1:n
            x[i,j]=(y[i,j]-real(sum(ss)))*dsuminv[i]
            for l = 1:k
                ss[l] =(ss[l]+x[i,j]*conj(d[l,i]))*conjb[l]
            end
        end
    end
    return x
end


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
    n=1000000
    x=rand(n)

    tic()
    cholsav=covchol(a,b,n)
    toc()
    tic()
    y=covprodut(x,cholsav...)
    toc()
    tic()
    z=covsolut(y,cholsav...)
    toc()
    println(norm(x-z)/norm(z))
end

#main()
