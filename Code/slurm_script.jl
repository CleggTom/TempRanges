using Pkg

Pkg.activate(".")
Pkg.update("FoodWebs")

using FoodWebs
using LinearAlgebra
using StatsBase, Polynomials
using JLD2

fw = FoodWebs

#get args
id = ARGS[1]
id_n = ARGS[2]
N = 10
C = 0.1

#function to get mass
get_M(com) = (com.R .^ [s.n for s = com.sp])

#1) GENERATE SPECIES POOL
N_pool = 100000
sp_vec = [fw.species(C) for i = 1:N_pool];

#2) ASSEMBLE COMMUNITIES
#set temperatures
N_T = 50
t_vec = range(0,1,length = N_T)

#compile
mc = fw.stable_metacommunity(sp_vec, 1, t_vec, T_range = 0.1, R = 43.0,
             psw_threshold = 0.8, max_draws = 10, verbose = false)
mc_prob = deepcopy(mc)
mc_random = deepcopy(mc)
fw.check_metacommunity(mc);


#3) DISPERSAL
# FoodWebs.K_dispersals!(mc1,10)
N_dispersal = 2
N_rep = 100
N_trials = 20

#allocate result arrays
psw = zeros(N_T, N_dispersal,2,N_trials)
params = Array{fw.GeneralisedParameters, 4}(undef, N_T, N_dispersal,2,N_trials)
bodysize = Array{Vector{Float64},3}(undef, N_T, N_dispersal,2)

#generate community
mc = fw.stable_metacommunity(sp_vec, N, t_vec, T_range = 0.1, R = 43.0,
             psw_threshold = 0.8, max_draws = 1000, verbose = false)
mc_prob = deepcopy(mc)
mc_random = deepcopy(mc)
fw.check_metacommunity(mc);

#simulate
for d = 1:N_dispersal
    #probabablistic
    for (i,c) = enumerate(mc_prob.coms)
        #itterate for given web structure
        for trial = 1:N_trials
            p = fw.generalised_parameters(c)
            J = similar(p.A)
            fw.generalised_jacobian!(J,p)
            psw[i,d,1,trial] = fw.max_real_eigval(J) 
            params[i,d,1,trial] = deepcopy(p)
        end
    end 

    #random
    for (i,c) = enumerate(mc_random.coms)
        #itterate for given web structure
        for trial = 1:N_trials
            p = fw.generalised_parameters(c)
            J = similar(p.A)
            fw.generalised_jacobian!(J,p)
            psw[i,d,2,trial] = fw.max_real_eigval(J) 
            params[i,d,2,trial] = deepcopy(p)
        end
    end

    bodysize[:,d,1] .= get_M.(mc_prob.coms)
    bodysize[:,d,2] .= get_M.(mc_random.coms)


    fw.multiple_dispersal!(mc_prob, p_dispersal = :p, d_dispersal = :p)
    fw.multiple_dispersal!(mc_random, p_dispersal = :r, d_dispersal = :r)

end

fn = join(["./data/results_",id,".jld2"])

save(fn, Dict("psw" => psw, "params" => params, "bodysize" => bodysize))