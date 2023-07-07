using Pkg
using Revise

Pkg.activate(".")

using FoodWebs
using LinearAlgebra
using StatsBase, Polynomials
using JLD2
using Distributions
using Dates

fw = FoodWebs

function exp_parameters(N::Int64, M::Int64)
    #exponent
    γ = rand(Uniform(0.8, 1.5), N, M) #[0.8, 1.5]
    λ = ones(N,N) # 1
    μ = rand(Uniform(1.0, 2.0), N, M) #[1.0, 2.0] 
    ϕ = rand(Uniform(0.0, 1.0), N, M) #[0.0, 1.0]
    ψ = rand(Uniform(0.5,1.1), N, M) #[0.5, 1.2]

    return [fw.ExponentialParameters(γ[:,i], λ, μ[:,i], ϕ[:,i], ψ[:,i]) for i = 1:M]
end

exp_parameters(com::fw.Community, M::Int64) = exp_parameters(com.N, M)

function get_jacobian(com::fw.Community, K)
    if com.N > 0
        J = zeros(com.N, com.N)
        λ_vals = Vector{Number}(undef, K)
        λ_vecs = Matrix{Number}(undef, K , com.N)
        s = fw.structural_parameters(com)

        for i = 1:K
            e = fw.generalised_parameters(com, K, f_ep = exp_parameters)
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

N_vec = [30]
C_vec = [0.2]
αd_vec = [0.75]

N_chunks = 10
chunk_size = 200
N_trials = 50

N_T = 50
t_vec = range(0,1,length = N_T)

#get masses function
get_M(com) = [com.R .^ n for n = com.n]

for N = eachindex(N_vec)
    for C = eachindex(C_vec)
        for αd = eachindex(αd_vec)
        
            
            println(Threads.nthreads())
            
            k = 0
            for c = 1:N_chunks
                
                psw = zeros(N_T,chunk_size,2,2)
                vecs = Array{Matrix{Number}, 4}(undef,N_T, chunk_size, 2,2)
                vals = Array{Vector{Number}, 4}(undef,N_T, chunk_size, 2,2)
                bodysize = Array{Vector{Float64},4}(undef,N_T, chunk_size, 2,2)


                Threads.@threads for r = 1:chunk_size

                        k += 1 
                        println("N: ",N, " C: ",C," r: ", r, " Thread: ", Threads.threadid(), " progress: ",k / chunk_size, " of chunk:", c)

                        mc = fw.stable_metacommunity(N_vec[N], C_vec[C], t_vec, T_range = 0.2, R = 43.0, psw_threshold = 0.7, verbose = false, vk = 100, max_draws = 2000, N_trials = N_trials, f_ep = exp_parameters)

                        mc_prob = deepcopy(mc)
                        mc_random = deepcopy(mc)

                        # get_jacobian(mc.coms, 1)

                        psw[:,r,1,1] .= psw[:,r,2,1] .= fw.proportion_stable_webs(mc, N_trials = N_trials, f_ep = exp_parameters);

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
                
                fn = join(["./data/psw_sensitvity/simulations_",N,"_",C,"_",αd,"_",c,".jld2"])
                save(fn, Dict("psw" => psw, "vecs" => vecs,"vals" => vals, "bodysize" => bodysize))
            
            end
            
            
       
        end
    end
end

