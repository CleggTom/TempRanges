using Pkg
using Revise

Pkg.activate(".")

using FoodWebs
using LinearAlgebra
using StatsBase, Polynomials
using JLD2

fw = FoodWebs

function get_jacobian(com::fw.Community, K)
    if com.N > 0
        J = zeros(com.N, com.N)
        λ_vals = Vector{Number}(undef, K)
        λ_vecs = Matrix{Number}(undef, K , com.N)
        s = fw.structural_parameters(com)

        for i = 1:K
            e = fw.generalised_parameters(com, K)
            fw.generalised_jacobian!(J, e[i])
            λ = eigen!(J)

            λ_vals[i] = λ.values[end]
            λ_vecs[i,:] .= λ.vectors[end,:]
        end

        d = Dict(:vals => λ_vals, :vecs => λ_vecs)

        return(d)
    else
        println(com)
        return(0)
    end
end

N_vec = [20]
C_vec = [0.2]
αd_vec = [0.75]

N_rep = 10
N_trials = 50

N_T = 50
t_vec = range(0,1,length = N_T)

#get masses function
get_M(com) = [com.R .^ n for n = com.n]

for N = eachindex(N_vec)
    for C = eachindex(C_vec)
        for αd = eachindex(αd_vec)
        
            psw = zeros(N_T,N_rep,2,2)
            vecs = Array{Matrix{Number}, 4}(undef,N_T, N_rep, 2,2)
            vals = Array{Vector{Number}, 4}(undef,N_T, N_rep, 2,2)
            bodysize = Array{Vector{Float64},4}(undef,N_T, N_rep, 2,2)

            println(Threads.nthreads())

            Threads.@threads for r = 1:N_rep
                    println("N: ",N, " C: ",C," r: ", r, " Thread: ", Threads.threadid())

                    mc = fw.stable_metacommunity(N_vec[N], C_vec[C], t_vec, T_range = 0.2, R = 43.0, psw_threshold = 0.8, verbose = false,max_draws = 2000, N_trials = 50)

                    mc_prob = deepcopy(mc)
                    mc_random = deepcopy(mc)

                    # get_jacobian(mc.coms, 1)

                    psw[:,r,1,1] .= psw[:,r,2,1] .= fw.proportion_stable_webs(mc, N_trials = N_trials);
                    bodysize[:,r,1,1] .= get_M.(mc_prob.coms)
                    bodysize[:,r,2,1] .= get_M.(mc_random.coms)


                    fw.multiple_dispersal!(mc_prob, p_dispersal = :r, d_dispersal = :p, αd = αd)
                    fw.multiple_dispersal!(mc_random, p_dispersal = :r, d_dispersal = :r, αd = αd)

                    N_T = 50
                    for c = 1:(N_T-1) # dont consider last community..
                        if mc_prob.coms[c].N > 0
                            λ = get_jacobian(mc_prob.coms[c], N_trials)
                            psw[c,r,1,2] = mean(real(λ[:vals]) .< 0)
                            vecs[c,r,1,2] = λ[:vecs]
                            vals[c,r,1,2] = λ[:vals]
                        end
                        
                        if mc_random.coms[c].N > 0
                            λ = get_jacobian(mc_random.coms[c], N_trials)
                            # print(mean(real(λ[:vals]) .< 0))
                            psw[c,r,2,2] = mean(real(λ[:vals]) .< 0)
                            vecs[c,r,2,2] = λ[:vecs]
                            vals[c,r,2,2] = λ[:vals]
                        end
                    end

                    bodysize[:,r,1,2] .= get_M.(mc_prob.coms)
                    bodysize[:,r,2,2] .= get_M.(mc_random.coms)
            end
            
            fn = join(["./data/psw_test/simulations_",N,"_",C,"_",αd,".jld2"])
            save(fn, Dict("psw" => psw, "vecs" => vecs,"vals" => vals, "bodysize" => bodysize))
       
        end
    end
end

