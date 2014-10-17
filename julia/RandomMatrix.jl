

# Based on Halko Martinsson & Tropp Algorithm 4.3
# Find range of matrix A using nl random vectors and
# with q power iterations.

function find_range_randomly(A::Matrix, num_lhs::Integer, q=1)
    m,n = size(A)
    Omega = randn(n, num_lhs)
    Y = A*Omega
    for _ in 1:q
        Y = A' * Y
        Y = A * Y
    end
    Q,R = qr(Y)
    Q
end


# Based on Halko Martinsson & Tropp Algorithm 5.1 for
# a randomized SVD

function find_svd_randomly(A::Matrix, num_lhs::Integer, q=2)
    Q = find_range_randomly(A, num_lhs, q)
    B = Q' * A
    u_b,w,v = svd(B)
    u = Q*u_b
    u,w,v
end

