import Pkg
Pkg.add("CUDAnative")
Pkg.add("CUDAdrv")
Pkg.add("CuArrays")
Pkg.add("BenchmarkTools")

using CUDAnative, CUDAdrv, CuArrays, Test, BenchmarkTools

function arrayAdd!(a, b, c)
    index = Int((blockIdx().x-1) * blockDim().x + threadIdx().x)
    stride = Int(blockDim().x * gridDim().x)
    for i in index:stride:length(a)
        @inbounds c[i] = a[i] + b[i]
    end
    return nothing
end

function bench1!(d_a, d_b, d_c)
    @sync begin
        @cuda blocks=ceil(Int, length(d_a)/1024) threads=1024 arrayAdd!(d_a, d_b, d_c)
    end
end

function bench2!(d_a, d_b, d_c)
    d_c .= d_b .+ d_a
end

M = 100000

h_a = round.(rand(Float32, M) * 100)
h_b = round.(rand(Float32, M) * 100)

d_a = CuArray(h_a)
d_b = CuArray(h_b)
d_c = similar(d_a)

bench1!(d_a, d_b, d_c)

h_c = Array(d_c)

@test isapprox(h_a + h_b, h_c)