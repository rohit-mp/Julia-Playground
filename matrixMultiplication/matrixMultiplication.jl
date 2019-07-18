import Pkg
Pkg.add("CUDAnative")
Pkg.add("CUDAdrv")
Pkg.add("CuArrays")
Pkg.add("BenchmarkTools")

using CUDAnative, CUDAdrv, CuArrays, Test, BenchmarkTools

function multiply(d_a, d_b, d_c)
    index = Int(blockDim().x * (blockIdx().x-1) + threadIdx().x)
    len = size(d_a, 1)
    x = ceil(Int, index/len)
    y = mod(index, len) + 1

    sum = 0

    for i in 1:len
        @inbounds sum += d_a[x,i] * d_b[i,y]        
    end
    
    @inbounds d_c[x,y] = sum
    
    return nothing
end

function bench_parallel(d_a, d_b, d_c)
    @cuda blocks=ceil(Int, sqrt(length(d_a))/1024) threads=1024 multiply(d_a, d_b, d_c)
    synchronize()
end

M = 2000

h_a = rand(Float32, M, M)
h_b = rand(Float32, M, M)
h_c = zeros(Float32, (M, M))

d_a = CuArray(h_a)
d_b = CuArray(h_b)
d_c = CuArray(h_c)

bench_parallel(d_a, d_b, d_c)

h_c = Array(d_c)
@test isapprox(h_c ,h_a * h_b)