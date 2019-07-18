import Pkg
Pkg.add("CUDAnative")
Pkg.add("CUDAdrv")
Pkg.add("CuArrays")
Pkg.add("BenchmarkTools")

using CUDAnative, CUDAdrv, CuArrays, Test, BenchmarkTools

function arraySum(a, res)
    index = blockDim().x * (blockIdx().x - 1) + threadIdx().x
    tidx = threadIdx().x
    
    shared_a = @cuStaticSharedMem(Float32, 1024)
    @inbounds shared_a[tidx] = a[index]
    
    ctr = 1
    while ctr < 1024
        if(tidx % (ctr*2) == 1)
            @inbounds shared_a[tidx] += shared_a[tidx+ctr]
        end
        ctr *= 2
        sync_threads()
    end
    
    if(tidx == 1)
        @atomic res[1] += shared_a[1]
    end
    return nothing
end

function bench_parallel(d_a, d_res)
    @sync @cuda blocks=ceil(Int, length(d_a)/1024) threads=1024 arraySum(d_a, d_res)
end

M = 10000000

h_a = rand(Float32, M)
d_a = CuArray(h_a)

h_res = 0.0f0

d_res = [0.0f0]
d_res = CuArray(d_res)

for i = 1:M
    h_res += h_a[i]
end

bench_parallel(d_a, d_res)

d_res = Array(d_res)
@test isapprox(d_res[1], h_res)