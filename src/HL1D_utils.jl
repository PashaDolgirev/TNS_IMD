# generates translationally invariant Hamiltonian terms
# we consider one, two, and three body Pauli terms 
# bulk of each generated term commutes with the string X1*X2*...*XN
function generate_custom_Hamiltonian(hl::HamiltonianLearner, coeff_list::Union{Vector{Float64}, Nothing}=nothing, range::Union{Int, Nothing}=2)
    N = length(hl.sites)
    H = AutoMPO()

    if isnothing(coeff_list)
        coeff_list = copy(hl.coeff_list)
    end

    num_ops = 24
    @assert length(coeff_list) == num_ops "Current implementation assumes $(num_ops) Hamiltonian terms" 

    ind_op = 1
    #single Pauli term
    for j in 1:N
        H .+= 2.0 * coeff_list[ind_op], "Sx", j
    end
    ind_op += 1

    #two-body Pauli terms
    for l in 1:range
        for j in 1:(N - l)
            H .+= 4.0 * coeff_list[ind_op], "Sx", j, "Sx", j + l
            H .+= 4.0 * coeff_list[ind_op + 1], "Sy", j, "Sy", j + l
            H .+= 4.0 * coeff_list[ind_op + 2], "Sz", j, "Sz", j + l
            H .+= 4.0 * coeff_list[ind_op + 3], "Sy", j, "Sz", j + l
            H .+= 4.0 * coeff_list[ind_op + 4], "Sz", j, "Sy", j + l
        end
        ind_op += 5
    end
    

    #three-body Pauli terms
    for l1 in 1:(range - 1)
        for l2 in (l1 + 1):range
            for j in 1:(N - l2)
                H .+= 8.0 * coeff_list[ind_op], "Sx", j, "Sx", j + l1, "Sx", j + l2

                H .+= 8.0 * coeff_list[ind_op + 1], "Sy", j, "Sx", j + l1, "Sy", j + l2
                H .+= 8.0 * coeff_list[ind_op + 2], "Sz", j, "Sx", j + l1, "Sz", j + l2
                H .+= 8.0 * coeff_list[ind_op + 3], "Sy", j, "Sx", j + l1, "Sz", j + l2
                H .+= 8.0 * coeff_list[ind_op + 4], "Sz", j, "Sx", j + l1, "Sy", j + l2

                H .+= 8.0 * coeff_list[ind_op + 5], "Sy", j, "Sy", j + l1, "Sx", j + l2
                H .+= 8.0 * coeff_list[ind_op + 6], "Sz", j, "Sz", j + l1, "Sx", j + l2
                H .+= 8.0 * coeff_list[ind_op + 7], "Sy", j, "Sz", j + l1, "Sx", j + l2
                H .+= 8.0 * coeff_list[ind_op + 8], "Sz", j, "Sy", j + l1, "Sx", j + l2

                H .+= 8.0 * coeff_list[ind_op + 9], "Sx", j, "Sy", j + l1, "Sy", j + l2
                H .+= 8.0 * coeff_list[ind_op + 10], "Sx", j, "Sz", j + l1, "Sz", j + l2
                H .+= 8.0 * coeff_list[ind_op + 11], "Sx", j, "Sz", j + l1, "Sz", j + l2
                H .+= 8.0 * coeff_list[ind_op + 12], "Sx", j, "Sy", j + l1, "Sy", j + l2
            end
            ind_op += 13
        end
    end

    @assert ind_op == (num_ops + 1) "Mismatch in counting operators"
    return H
end

function generate_Hamiltonian(hl::HamiltonianLearner, coeff_list::Union{Vector{Float64}, Nothing}=nothing, flag_diag=true, maxdim=[10, 20, 100, 100, 200])
    if isnothing(coeff_list)
        coeff_list = copy(hl.coeff_list)
    end

    @assert length(coeff_list) == length(hl.op_list) "Number of operators and coefficients are inconsistent"

    H = generate_custom_Hamiltonian(hl, coeff_list)
    hl.H = MPO(H, hl.sites)

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
    return G_mat
end


function find_optimal_eigenvec(V::Vector{Vector{Float64}})
    d = length(V)

    # Define custom weights for penalizing terms
    custom_weights = [0.0; 
                         ones(Float64, 5);#2-Pauli nn
                         4.0 * ones(Float64, 5); #2-Pauli nnn
                         4.0 * ones(Float64, 5); #3-Pauli
                         8.0 * ones(Float64, 8)] #3-Pauli skewed
    custom_weights[4] = 0.5  # prefer ZZ
    custom_weights[14] = 1.5  # prefer ZXZ

    # Combine V[i] with weights x[i] to get vector in degenerate subspace
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

    # Initial guess
    x0 = ones(Float64, length(V))

    result = optimize(cost_vec, x0, NelderMead())
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
        if abs(eigvals[n] - λ_min) < 1e-12
            push!(V, eigvecs[:, n])
        end
    end

    if length(V) == 1
        v_min = V[1]
    else
        v_min = find_optimal_eigenvec(V)
    end

    H = generate_custom_Hamiltonian(hl, v_min)

    E_psi = real(inner(hl.psi', MPO(H, hl.sites), hl.psi))
    if  E_psi> 0
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