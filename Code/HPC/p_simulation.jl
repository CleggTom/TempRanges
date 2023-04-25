start = time()

using Pkg

Pkg.activate(".")

using FoodWebs
using LinearAlgebra
using StatsBase
using JLD2

fw = FoodWebs

#get args
id = ARGS[1]
id_n = ARGS[2]
save_loc = ARGS[3]


i = parse(Int64, id_n) - 1

x,y = i ÷ 5 , i % 5

N = [5,10,15,20,25][x]
C = [0.1, 0.15, 0.2, 0.25, 0.3][y]

println("generate species")
#1) GENERATE SPECIES POOL
N_pool = 2000
sp_vec = [fw.species(C) for i = 1:N_pool];

println("Assemble community")
#2) ASSEMBLE COMMUNITIES
#set temperatures
N_T = 50
T_vec = range(0,1,length = N_T)
fw.stable_metacommunity_p(sp_vec, 1, T_vec)

println("Dispersal")
#3) DISPERSAL
# FoodWebs.K_dispersals!(mc1,10)
N_dispersal = 2
N_rep = 8

# stability = zeros(N_T, N_rep, N_dispersal, 2)
jacobians = Array{Any,4}(undef,N_T, N_rep, N_dispersal, 2)
params = Array{Any,4}(undef,N_T, N_rep, N_dispersal, 2)

p = [0]

for r = 1:N_rep
    p[1] += 1
    println(p[1])

    N_pool = 10000
    sp_vec = [fw.species(C) for i = 1:N_pool];

    #generate inial communities
    mc = fw.stable_metacommunity_p(sp_vec, N, T_vec, N_trials = 1000)
    mc_prob = deepcopy(mc)
    mc_rand = deepcopy(mc)
    for d = 1:N_dispersal 
        #jacobians
        jacobians[:, r, d, 1] .= fw.generalised_jacobian.([com.p for com = mc_prob.coms])
        jacobians[:, r, d, 2] .= fw.generalised_jacobian.([com.p for com = mc_rand.coms])

        #calculate stability
        params[:, r, d, 1] .= [com.p for com = mc_prob.coms]
        params[:, r, d, 2] .= [com.p for com = mc_rand.coms]


        fw.multiple_dispersal!(mc_prob, p_dispersal = :p, d_dispersal = :p)
        fw.multiple_dispersal!(mc_rand, p_dispersal = :r, d_dispersal = :r)
    end
end

println("saving")

fn = join([save_loc,"/results_",id,"_",id_n,".jld2"])
println(fn)

save(fn, Dict("jacobians" => jacobians, "params" => params, "R" => 42.0))

e = time()
print("start: ", start, " end: ", e, " time taken: ", e - start )