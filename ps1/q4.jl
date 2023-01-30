#   Problem 4 : Estimation - BLP
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

using Plots, DataFrames, CSV, GLM
using Optim, Distributions, Random, ForwardDiff
using LinearAlgebra,StatsFuns

# load in csv
df = DataFrame(CSV.File("../data/ps1_ex4.csv"));

# simulate individual taste shocks from N(μ,Σ)
draw_sim = function(μ, Σ, N) # return N x L matrix
    # draw shocks
    v = rand(MvNormal(μ, Σ), N)
    
    return v
end

# data params
n_markets = 100

n_sim = 50;
Random.seed!(6789998212);

# get data
x_t = []
for t in 1:n_markets 
   push!(x_t, Array(df[df[!,:market].==t, [:p, :x]]))
end

s_t = []
for t in 1:n_markets 
    push!(s_t, Array(df[df[!,:market].==t, [:shares]]))
end

z_t = []
for t in 1:n_markets 
    push!(z_t, Array(df[df[!,:market].==t, [:z1, :z2, :z3, :z4, :z5, :z6, :x]]))
end

z_jt = Array(df[df[!,:market] .<= n_markets,[:z1, :z2, :z3, :z4, :z5, :z6, :x]]);

v = draw_sim([0;0], [1 0;0 1], n_sim);

#   Part 1: BLP
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡

#   Inner loop
#   ============
# 
#   get_shares calculates the shares of each product in a particular market t.
#   \delta should be a vector of length J; x should be a matrix of size J \times
#   2; and v should be a matrix of size L x 2.
# 
#   delta_contraction iterates the \delta_{jt} in a particular market t. \delta
#   should be a vector of length J; x should be a matrix of characteristics of
#   size J x 2; s should be a vector of observed shares with length J; v should
#   be a vector of length L x 2.
# 
#   market_iterate performs the contraction over each t markets, it recoves
#   \delta, which is a vector of length J \times T.

# get shares in a market given some fixed gamma and delta
get_shares = function(δ, Γ, x, v) # returns J length vector
    # we want to get share_{jt} using simulated values of $v_i$ (drawn above)
    # shares should be vector of length J
    numerator = exp.(δ .+ x * Γ * v)
    adj = maximum(numerator, dims = 1)
    denominator = sum((numerator ./ adj), dims = 1) .+ (1 ./ adj)
    shares = sum((numerator ./ adj) ./ denominator, dims = 2) ./ size(v)[2]
    
    return shares
end

# inner loop: contraction to find δ
delta_contraction = function(δ₀, Γ, s, x, v, tol = 1e-14, max_iter = nothing) # returns J length vector

    # here δ is a vector of length J
    δ = δ₀
    err = 1000
    n = 0
    maxed_iter = false
    
    while (err > tol) && (maxed_iter === false)
        δ_old = δ
        
        # update delta
        δ = δ_old + log.(s) - log.(get_shares(δ_old, Γ, x, v))
        
        # difference 
        err = maximum(abs.(δ - δ_old)) 
        
        # (optional) max iterations block
        n += 1
        if max_iter !== nothing
            maxed_iter = (n == max_iter)
        end
    end
    
    return δ
end

# iterate over each market
market_iterate = function(Γ, s_t, x_t, v, tol = 1e-14, max_iter = nothing) # returns T length vector of J length vectors
   
    δ = []
    for t in 1:size(s_t)[1]
        s = s_t[t]
        x = x_t[t]
        δ₀ = ones(size(s)[1])
        push!(δ, delta_contraction(δ₀, Γ, s, x, v, tol, max_iter) ) 
    end
    return δ
end

#   Outer loop
#   ============
# 
#   residuals does IV-GMM using the provided weighting matrix. zjt should be a
#   matrix of Z excluded and included intruments of size JT \times Z. Returns
#   linear parameters (vector of length 2) and :\xi{jt}$ residuals (vector of
#   length JT)
# 
#   gmm_objective Reads in JT-length vector \xijt and JT \times Z matrix zjt.
#   Stacks JT sample moments (size of instrument vector, Z) into g, returns
#   objective (scalar).
# 
#   outer_loop is the wrapper for the objective function for the solver. It
#   takes in the parameter guess \theta_2, data, and weighting matrix, then
#   performs the inner loop, returning the GMM objective.

# returns residuals for a given δ, estimates linear parameters given instruments
resid = function(δ_jt, x_jt, z_jt, W)
    # iv-gmm
    θ₁ = inv(x_jt' * (z_jt * W * z_jt') * x_jt) * (x_jt' * (z_jt * W * z_jt') * δ_jt)
    ξ_jt = δ_jt - x_jt * θ₁
    
    return ξ_jt, θ₁ 
    
end

# calculates gmm objective for outer loop
function gmm_objective(ξ_jt, z_jt, W)   
    # empirical moments, weighting matrix
    g = (ξ_jt' * z_jt)
    
    # gmm objective
    G = g * W * g'
    
    return G
end

# performs outer loop
function outer_loop(θ₂, s_t, x_t, z_jt, v, W, tol = 1e-14, max_iter = nothing)
    # Pass through guess
    Γ = [θ₂[1] 0 ; θ₂[2] θ₂[3]] # lower triangular
    
    # Perform inner loop
    δ = market_iterate(Γ, s_t, x_t, v, tol, max_iter)
    
    # stack markets into JT length matrix
    δ_jt = vec(reduce(hcat,δ)) 
    x_jt = reduce(vcat,x_t)
    
    # intermediate step to retrieve estimates
    ξ_jt, θ₁ = resid(δ_jt, x_jt, z_jt, W)
    
    # gmm step
    G = gmm_objective(ξ_jt, z_jt, W)
    
    return G
end


#   2-step GMM
#   ============

# this will return θ₁ & θ₂ for any given weighting matrix
function gmm_step(params0, s_t, x_t, z_jt, v, w, tol=1e-14, max_iter=nothing)
    f(θ₂) = outer_loop(θ₂, s_t, x_t, z_jt, v, w, tol, max_iter)
    o = Optim.optimize(f, params0, BFGS(), Optim.Options(show_trace = true, show_every = 10))
    show(o)
    # step 1.5: recover θ₁ from θ₂
    θ₂ = o.minimizer
    Γ = [θ₂[1] 0 ; θ₂[2] θ₂[3]]
    δ = market_iterate(Γ, s_t, x_t, v, tol, max_iter)
    δ_jt = vec(reduce(hcat,δ)) 
    x_jt = reduce(vcat,x_t)
    ξ_jt, θ₁ = resid(δ_jt, x_jt, z_jt, w)
    println(θ₁, Γ)
    return θ₁, θ₂, ξ_jt
end

# this will return the parameters estimated using the efficient weighting matrix
function two_step(params0, s_t, x_t, z_jt, v, tol=1e-14, max_iter=nothing)
    
    # step 1: use the TSLS weighting matrix to get a consistent estimator of θ
    w = inv(z_jt' * z_jt)
    θ₁, θ₂, ξ_jt = gmm_step(params0, s_t, x_t, z_jt, v, w)
    
    # step 1.5: get the efficient weighting matrix 
    w = inv((ξ_jt .* z_jt)' * (ξ_jt .* z_jt))
    
    # step 2: gmm again with the efficient weighting matrix, update starting values
    θ₁, θ₂, ξ_jt = gmm_step(θ₂, s_t, x_t, z_jt, v, w)
    
    return θ₁, θ₂, ξ_jt, w
end

params0 = ones(3)
θ₁, θ₂, ξ_jt, w = two_step(params0, s_t, x_t, z_jt, v);

# test 
show(θ₁)
show(θ₂)
show(ξ_jt)
show(w)

Γ = [θ₂[1] 0 ; θ₂[2] θ₂[3]]
δ = market_iterate(Γ, s_t, x_t, v)
δ_jt = vec(reduce(hcat,δ)) 

S_t = []
for t in 1:size(x_t)[1]
    push!(S_t, get_shares(δ[t], Γ, x_t[t], v))
end
S = mean(reduce(hcat,S_t), dims = 2)

s = mean(reduce(hcat,s_t), dims = 2)

#print(θ₁, θ₂)
maximum(reduce(hcat,S_t) - reduce(hcat,s_t))


#reshape(ξ_jt,6,Int64(size(ξ_jt)[1] / 6))

#   Gradient Appendix
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡
# 
#   Gradient
#   ==========
# 
#   Estimator is \nabla G(\theta) = 2(Z'J_\theta)'W(Z'\xi(\theta)) -Z is JT
#   \times Z matrix if instruments
# 
#   -\xi is JT \times 1 matrix of unobserved mean utilities
# 
#   -W is Z \times Z
# 
# :$
# 
#   W = \left[(\xi(\theta) \circ Z)' (\xi(\theta) \circ Z) \right]^{-1}:$
# 
#   -J_\theta is JT \times 3
# 
# :$
# 
#   J\theta = -f\xi^{-1} f_\theta:$
# 
#     •  For each t, f_\xi is a J \times J matrix: \left\{\frac{\partial
#        s_{ij}}{\partial \xi_k}\right\}_{j,k}
# 
# :$
# 
#   \frac{\partial s{ij}}{\partial \xik} = -s{ij}s{ik}, \quad \frac{\partial
#   s{ij}}{\partial \xij} = s{ij}(1-s{ij}) :$
# 
#     •  For each t, f_\theta is a J \times 3 matrix:
# 
# :$
# 
#   \begin{bmatrix} s{ij}s{i0}\left(pj - \sumk s{ik}pk \right)\nu{1i} &
#   s{ij}s{i0}\left(xj - \sumk s{ik}xk \right)\nu{1i} & s{ij}s{i0}\left(xj -
#   \sumk s{ik}xk \right)\nu{2i} \end{bmatrix}{j} :$
# 
#   All matrices are stacked J \times \cdot over T markets

# Steps to calculate gradient...
# have data s_t, x_t, z_t
# first step will return θ           -> Γ  x
# 1. run market_iterate() with Γ     -> δ  x
# 2. run resid() with δ              -> ξ x
# 3. calculate W with ξ and Z        -> W x
# 4. calculate J with δ and Γ*       -> J  x
# *(for each i, calculate s_ij vector, do elementwise mult with p_j, v, and sum to get f_xi loop through j,k for f_theta)
# 5. run gradient() with J, W, ξ, Z  -> ∇

# helper function in gradient call: for each market get Jacobian of ξ(θ)
function jacobian_xi(δ, Γ, x, v)
    # need individual shares
    numerator = exp.(δ .+ x * Γ * v)
    adj = maximum(numerator, dims = 1)
    denominator = sum((numerator ./ adj), dims = 1) .+ (1 ./ adj)
    shares = (numerator ./ adj) ./ denominator # J x L
    
    # calculate partials of f(θ) = s - S(ξ,θ), denoted fξ and fθ
    fξ_store = []
    fθ_store = [] 
    for i = 1:size(v)[2]
        s_i = shares[:,i]
        s_i0 = 1 - sum(s_i)
        v_i = v[:,i]
        
        fξ_i = - s_i * s_i' + diagm(s_i)
        
        fθ₁ = s_i .* (x[:,1] .- (s_i' * x[:,1])) .* v[1]
        fθ₂ = s_i .* (x[:,2] .- (s_i' * x[:,2])) .* v[1]
        fθ₃ = s_i .* (x[:,2] .- (s_i' * x[:,2])) .* v[2]
        fθ_i = hcat(fθ₁, fθ₂, fθ₃)
        
        push!(fξ_store, fξ_i)
        push!(fθ_store, fθ_i)        
    end
    
    # calculate Jacobian
    J = -1 .* inv(mean(fξ_store)) * mean(fθ_store)
   
    return J
end
    
function gmm_gradient!(θ₂, s_t, x_t, z_jt, v, W, ∇, tol = 1e-12, max_iter = nothing)
    # Pass through guess
    Γ = [θ₂[1] 0 ; θ₂[2] θ₂[3]] # lower triangular
    
    println(1, Γ)
    
    # Recover model objects from estimates parameters: Γ, and data: s_t, x_t, z_t, and v (simulated)    
    # δ(θ)
    δ = market_iterate(Γ, s_t, x_t, v, tol, max_iter)
    
    println(2,δ)
    
    # ξ(θ)
    δ_jt = vec(reduce(hcat,δ)) 
    x_jt = reduce(vcat,x_t)
    ξ_jt = resid(δ_jt, x_jt, z_jt, W)[1]
    ξ_t = reshape(ξ_jt, 6, Int64(size(ξ_jt)[1] / 6))
    
    println(3, ξ_t)
    
    # Analytic matrices
    # Jacobian
    J_t = []
    for t = 1:size(x_t)[1]
        push!(J_t, jacobian_xi(δ[t], Γ, x_t[t], v))
    end
    # J = reduce(vcat, J_t) # flatten to JT x 3 matrix
    
    println(4, mean(J_t))
  
    # Calculate gradient
    ∇_t = []
    for t in 1:size(s_t)[1]
        push!(∇_t,  2 .* (z_t[t]' * J_t[t])' * W * (z_t[t]' * ξ_t[:,t]))
    end
    ∇ .= mean(∇_t)
    
    print(5,∇)

    return ∇
end

# using ForwardDiff
# ForwardDiff.gradient(f,θ₂)
# ∇ = ones(3)
# gmm_gradient!(θ₂, s_t, x_t, z_jt, v, w, ∇, tol, max_iter)

# define g 
# g!(G,θ₂) = gmm_gradient!(θ₂, s_t, x_t, z_jt, v, w, G, tol, max_iter)

using NBInclude
nbexport("q4.jl", "q4.ipynb")