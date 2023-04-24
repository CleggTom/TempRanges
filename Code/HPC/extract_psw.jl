using Pkg

Pkg.activate(".")

using FoodWebs
using LinearAlgebra
using StatsBase
using JLD2

#exclude NaNs
mean_nan(x) = mean(filter(y -> !isnan(y), x))
mean_nan(d,dim) = mapslices(mean_nan, d, dims = dim)

#loop over files and extract psw
files = readdir(ARGS[1])
filter!(x -> endswith(x, ".jld2"), files)

res = Vector{Any}(undef, length(files))

for (i,f) in enumerate(files)
    if endswith(f, ".jld2")
        fn = join(["/albedo/work/user/thcleg001/Projects/TempRanges/Data/test_sims/",f])
        r = load(fn)

        res[i] = mean_nan(r["psw"] .< 0, 4)[:,:,:,1]
    end
end

psw = cat(res..., dims = 4)
save(ARGS[2], Dict("psw" => psw))