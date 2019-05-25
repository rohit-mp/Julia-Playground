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
    @sync begin #not sure why I'm using this
        @cuda blocks=ceil(Int, length(d_a)/1024) threads=1024 arrayAdd!(d_a, d_b, d_c)
    end
end
#=The most interesting part of this is the call to CuArrays.@sync.
The CPU can assign jobs to the GPU and then go do other stuff (such as assigning more jobs to the GPU) while 
the GPU completes its tasks. Wrapping the execution in a CuArrays.@sync block will make the CPU block until the 
queued GPU tasks are done, similar to how Base.@sync waits for distributed CPU tasks.
Without such a synchronization, you'd be measuring the time takes to launch the computation, 
not the time to perform the computation. But most of the time you don't need to synchronize explicitly: 
many operations, like copying memory from the GPU to the CPU, implicitly synchronize execution.=#

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
#synchronize() if you're printing anything in the kernel

h_c = Array(d_c)

@test isapprox(h_a + h_b, h_c)