using Base.Test
include ("NoiseAnalysis.jl")

# Test that noise covariance model noise can correctly
# whiten random data of lenght n

function test_whitening(noise::NoiseModel, n::Integer)
    r = noise_covariance(noise, n)
    R = toeplitz(r)
    L = chol(R, :L)
    W = inv(L)

    data = rand(n)+1.0
    true_w = W * data
    w = whiten(noise, data)
    for i=1:n
        @test_approx_eq_eps w[i] true_w[i] 1e-4
    end
end

function test_1component(b1)
    b = [b1]
    a = [1.0, 0.2]
    nm = NoiseModel(b, a, 100)
    test_whitening(nm, 5)
    test_whitening(nm, 100)
    test_whitening(nm, 200)
    test_whitening(nm, 1000)
end

test_1component(0.997)
test_1component(0.95)
test_1component(0.53)

function test_3components(b)
    @assert length(b) == 3
    a = [1.0, 0.5, 0.5, 1.0]

    nm = NoiseModel(b, a, 100)
    test_whitening(nm, 5)
    test_whitening(nm, 100)
    test_whitening(nm, 200)
    test_whitening(nm, 1000)
end

test_3components([.99, .95, .80])
test_3components([.99, .95+.1im, .95-.1im])
