# generates translationally invariant Hamiltonian terms
# we consider one, two, and three body Pauli terms 
# bulk of each generated term commutes with the string X1*X2*...*XN
function generate_custom_Hamiltonian(hl::HamiltonianLearner, coeff_list::Union{Vector{Float64}, Nothing}=nothing)
    N = length(hl.sites)
    H = AutoMPO()

    if isnothing(coeff_list)
        coeff_list = copy(hl.coeff_list)
    end

    ind_op = 1
    #single Pauli term
    for j in 1:N
        H .+= 2.0 * coeff_list[ind_op], "Sx", j
    end
    ind_op += 1

    #two-body Pauli terms
    for j in 1:(N - 1)
        H .+= 4.0 * coeff_list[ind_op], "Sx", j, "Sx", j + 1
        H .+= 4.0 * coeff_list[ind_op + 1], "Sy", j, "Sy", j + 1
        H .+= 4.0 * coeff_list[ind_op + 2], "Sz", j, "Sz", j + 1
        H .+= 4.0 * coeff_list[ind_op + 3], "Sy", j, "Sz", j + 1
        H .+= 4.0 * coeff_list[ind_op + 4], "Sz", j, "Sy", j + 1
    end
    ind_op += 5
    
    #three-body Pauli terms
    #bulk terms
    for j in 1:(N - 2)
        H .+= 8.0 * coeff_list[ind_op], "Sx", j, "Sx", j + 1, "Sx", j + 2
        H .+= 8.0 * coeff_list[ind_op + 1], "Sy", j, "Sx", j + 1, "Sy", j + 2
        H .+= 8.0 * coeff_list[ind_op + 2], "Sz", j, "Sx", j + 1, "Sz", j + 2
        H .+= 8.0 * coeff_list[ind_op + 3], "Sy", j, "Sx", j + 1, "Sz", j + 2
        H .+= 8.0 * coeff_list[ind_op + 4], "Sz", j, "Sx", j + 1, "Sy", j + 2
    end
    #corresponding boundary terms
    # H .+= 4.0 * coeff_list[ind_op], "Sx", 1, "Sx", 2
    # H .+= 4.0 * coeff_list[ind_op], "Sx", N - 1, "Sx", N
    # H .+= 4.0 * coeff_list[ind_op + 1], "Sx", 1, "Sy", 2
    # H .+= 4.0 * coeff_list[ind_op + 1], "Sy", N - 1, "Sx", N
    # H .+= 4.0 * coeff_list[ind_op + 2], "Sx", 1, "Sz", 2
    # H .+= 4.0 * coeff_list[ind_op + 2], "Sz", N - 1, "Sx", N
    # H .+= 4.0 * coeff_list[ind_op + 3], "Sx", 1, "Sz", 2
    # H .+= 4.0 * coeff_list[ind_op + 3], "Sy", N - 1, "Sx", N
    # H .+= 4.0 * coeff_list[ind_op + 4], "Sx", 1, "Sy", 2
    # H .+= 4.0 * coeff_list[ind_op + 4], "Sz", N - 1, "Sx", N
    ind_op += 5

    #boundary terms - single Pauli
    H .+= 2.0 * coeff_list[ind_op], "Sx", 1
    H .+= 2.0 * coeff_list[ind_op + 1], "Sx", N
    H .+= 2.0 * coeff_list[ind_op + 2], "Sy", 1
    H .+= 2.0 * coeff_list[ind_op + 3], "Sy", N
    H .+= 2.0 * coeff_list[ind_op + 4], "Sz", 1
    H .+= 2.0 * coeff_list[ind_op + 5], "Sz", N
    ind_op += 6

    #boundary terms - two Paulis
    H .+= 4.0 * coeff_list[ind_op], "Sx", 1, "Sx", 2
    H .+= 4.0 * coeff_list[ind_op + 1], "Sx", 1, "Sy", 2
    H .+= 4.0 * coeff_list[ind_op + 2], "Sx", 1, "Sz", 2
    H .+= 4.0 * coeff_list[ind_op + 3], "Sy", 1, "Sx", 2
    H .+= 4.0 * coeff_list[ind_op + 4], "Sy", 1, "Sy", 2
    H .+= 4.0 * coeff_list[ind_op + 5], "Sy", 1, "Sz", 2
    H .+= 4.0 * coeff_list[ind_op + 6], "Sz", 1, "Sx", 2
    H .+= 4.0 * coeff_list[ind_op + 7], "Sz", 1, "Sy", 2
    H .+= 4.0 * coeff_list[ind_op + 8], "Sz", 1, "Sz", 2
    ind_op += 9

    H .+= 4.0 * coeff_list[ind_op], "Sx", N - 1, "Sx", N
    H .+= 4.0 * coeff_list[ind_op + 1], "Sx", N - 1, "Sy", N
    H .+= 4.0 * coeff_list[ind_op + 2], "Sx", N - 1, "Sz", N
    H .+= 4.0 * coeff_list[ind_op + 3], "Sy", N - 1, "Sx", N
    H .+= 4.0 * coeff_list[ind_op + 4], "Sy", N - 1, "Sy", N
    H .+= 4.0 * coeff_list[ind_op + 5], "Sy", N - 1, "Sz", N
    H .+= 4.0 * coeff_list[ind_op + 6], "Sz", N - 1, "Sx", N
    H .+= 4.0 * coeff_list[ind_op + 7], "Sz", N - 1, "Sy", N
    H .+= 4.0 * coeff_list[ind_op + 8], "Sz", N - 1, "Sz", N

    return H
end

function generate_Hamiltonian(hl::HamiltonianLearner, coeff_list::Union{Vector{Float64}, Nothing}=nothing, flag_diag=true)
    if isnothing(coeff_list)
        coeff_list = copy(hl.coeff_list)
    end

    @assert length(coeff_list) == length(hl.op_list) "Number of operators and coefficients are inconsistent"

    H = generate_custom_Hamiltonian(hl, coeff_list)
    hl.H = copy(MPO(H, hl.sites))

    if flag_diag
        psi0 = deepcopy(hl.psi)#randomMPS(hl.sites, linkdims=10)
        maxdim_schedule = [10, 20, 100, 200, 400, 800, 800, 800]
        sweeps = Sweeps(length(maxdim_schedule))
        maxdim!(sweeps, maxdim_schedule...)
        cutoff!(sweeps, 1e-10)

        E_GS, ψ_GS = dmrg(hl.H, psi0, sweeps)
        hl.ψ_GS = copy(ψ_GS)
        hl.E_GS = E_GS
        hl.Fidelity = abs2(inner(ψ_GS', hl.psi))

        println("Fidelity = $(hl.Fidelity)")
        println("GS energy = $(E_GS)")
    end
    return E_GS
end

function generate_correlation_matrix(hl::HamiltonianLearner)

    n_ops = length(hl.op_list)
    psi_list = Vector{MPS}(undef, n_ops)
    G_mat = Matrix{Float64}(undef, n_ops, n_ops)
    expectation_vals = Vector{Float64}(undef, n_ops)

    for j in 1:n_ops
        OPj = MPO(hl.op_list[j], hl.sites)
        ψj = OPj * hl.psi
        psi_list[j] = ψj
        expectation_vals[j] = real(inner(hl.psi', OPj, hl.psi))
    end

    for i in 1:n_ops, j in 1:n_ops
        G_mat[i, j] = real(inner(psi_list[i]', psi_list[j])) - expectation_vals[i] * expectation_vals[j]
    end

    G_mat = 0.5 * (G_mat + G_mat')
    hl.G_mat = copy(G_mat)
    hl.exp_vals = copy(expectation_vals)
    return G_mat
end


function find_optimal_eigenvec(V::Vector{Vector{Float64}}, exp_vals::Vector{Float64})
    d = length(V)

    custom_weights = [0.0; 
                         ones(Float64, 5);#2-Pauli nn
                         4.0 * ones(Float64, 5); #3-Pauli
                         10.0 * ones(Float64, 6); #boundary: single Pauli
                         1000.0 * ones(Float64, 18); #boundary: two Paulis
                         ] 
    custom_weights[4] = -100.5  # prefer ZZ
    custom_weights[9] = -1000.5  # prefer ZXZ
    custom_weights[20] = -1000.5 # prefer ZXZ
    custom_weights[33] = -1000.5 # prefer ZXZ

    function get_vec(x)
        v = zero(V[1])
        for i in 1:d
            v .+= x[i] * V[i]
        end
        return v / norm(v)
    end

    # Cost function to minimize (weighted L2 norm of resulting vector)
    function cost_vec(x)
        v = get_vec(x)
        return sum((v .^ 2) .* custom_weights)
    end

    x0 = ones(Float64, length(V))

    result = optimize(cost_vec, x0, NelderMead())
    x_opt = Optim.minimizer(result)

    function cost_vec_2(x)
        v = get_vec(x)
        return sum((v .^ 2) .* custom_weights) + 1000.0 * sum(exp_vals .* v)
    end

    result = optimize(cost_vec_2, x_opt, NelderMead())
    x_opt = Optim.minimizer(result)

    v_opt = get_vec(x_opt)
    return v_opt
end


function LearnHamiltonian_corr_mat(hl::HamiltonianLearner)
    G_mat = generate_correlation_matrix(hl)

    eigvals, eigvecs = eigen(G_mat)
    λ_min = eigvals[1]

    V = Vector{Vector{Float64}}()

    for n in 1:length(eigvals)
        if abs(eigvals[n] - λ_min) < 1e-10
            push!(V, eigvecs[:, n])
        end
    end

    if length(V) == 1
        v_min = V[1] / norm(V[1])
    else
        v_min = find_optimal_eigenvec(V, hl.exp_vals)
    end

    H = generate_custom_Hamiltonian(hl, v_min)

    E_psi = real(inner(hl.psi', MPO(H, hl.sites), hl.psi))
    if  E_psi > 0
        v_min = -v_min
        E_psi = -E_psi
    end

    println("Degeneracy = $(length(V))")
    _ = generate_Hamiltonian(hl, v_min)
    hl.λ_min = λ_min
    hl.E_psi = E_psi
    hl.coeff_list = copy(v_min)
    return λ_min, v_min
end