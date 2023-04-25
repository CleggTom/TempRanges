using Pkg
using Revise

Pkg.activate(".")
Pkg.update("FoodWebs")

using FoodWebs
using LinearAlgebra
using StatsBase, Polynomials
using JLD2

fw = FoodWebs

#1) GENERATE SPECIES POOL
N_pool = 100000
sp_vec = [fw.species(0.1) for i = 1:N_pool];

#2) ASSEMBLE COMMUNITIES
#set temperatures
N_T = 50
t_vec = range(0,1,length = N_T)

function generate_mc(sp_vec; verbose = false, vk = 100)
    #generate metacommunty
    mc = fw.stable_metacommunity(sp_vec, 20, t_vec, T_range = 0.1, R = 43.0,
             psw_threshold = 0.8, max_draws = 2000, verbose = verbose, vk = vk)

    mc_prob = deepcopy(mc)
    mc_random = deepcopy(mc)

    fw.check_metacommunity(mc);
    
    return mc_prob,mc_random
end

get_M(com) = (com.R .^ [s.n for s = com.sp])

#3) DISPERSAL
# FoodWebs.K_dispersals!(mc1,10)
N_dispersal = 2
N_rep = 100
N_trials = 20

res = zeros(N_T, N_dispersal,N_rep,2,N_trials)
params = Array{fw.GeneralisedParameters, 5}(undef, N_T, N_dispersal,N_rep,2,N_trials)
bodysize = Array{Vector{Float64},4}(undef, N_T, N_dispersal,N_rep,2)

println(Threads.nthreads())

Threads.@threads for r = 1:N_rep
    println("r: ", r, " Thread: ", Threads.threadid())
    mc_prob,mc_random = generate_mc(sp_vec, verbose = false)
    
    
    for d = 1:N_dispersal
        #probabablistic
        for (i,c) = enumerate(mc_prob.coms)
            #itterate for given web structure
            for trial = 1:N_trials
                p = fw.generalised_parameters(c)
                J = similar(p.A)
                fw.generalised_jacobian!(J,p)
                res[i,d,r,1,trial] = fw.max_real_eigval(J) 
                params[i,d,r,1,trial] = deepcopy(p)
            end
        end
        
        #random
        for (i,c) = enumerate(mc_random.coms)
            #itterate for given web structure
            for trial = 1:N_trials
                p = fw.generalised_parameters(c)
                J = similar(p.A)
                fw.generalised_jacobian!(J,p)
                res[i,d,r,2,trial] = fw.max_real_eigval(J) 
                params[i,d,r,2,trial] = deepcopy(p)
            end
        end
        
        bodysize[:,d,r,1] .= get_M.(mc_prob.coms)
        bodysize[:,d,r,2] .= get_M.(mc_random.coms)


        fw.multiple_dispersal!(mc_prob, p_dispersal = :p, d_dispersal = :p)
        fw.multiple_dispersal!(mc_random, p_dispersal = :r, d_dispersal = :r)

        if d % 100 == 0
            print(i)
        end
    end
end

save("./data/simulations.jld2", Dict("res" => res, "params" => params, "bodysize" => bodysize))