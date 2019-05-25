import Pkg
Pkg.add("CUDAnative")
Pkg.add("CUDAdrv")
Pkg.add("CuArrays")
Pkg.add("BenchmarkTools")

using CUDAnative, CUDAdrv, CuArrays, Test, BenchmarkTools

function arraySum!(a)
    index = Int(blockDim().x * (blockIdx().x - 1) + threadIdx().x)
    ctr = 2
    while ctr <= 1024
        if(index % ctr == 1)
            @inbounds a[index] += a[index+(ctr>>>1)]
        end
        ctr = ctr<<1
        sync_threads()
    end
    
    if(Int(threadIdx().x) == 1 && index != 1)
        @inbounds @atomic a[1] += a[index]
    end
    return nothing
end

function bench!(d_a)
    @sync @cuda blocks=ceil(Int, M/1024) threads=1024 arraySum!(d_a)
end

M = 10000000

h_a = rand(Float32, M)
d_a = CuArray(h_a)

result = 0
for i = 1:M
    result += h_a[i]
end

bench!(d_a)

h_b = Array(d_a)

@test isapprox(h_b[1], result)