function generate_2qubit_unitary(θ_vec::Vector{Float64})
    if length(θ_vec) != 15
        @error "Provided circuit parameters don't encode 15 param 2-qubit unitary"
    end
    I, X, Y, Z = [1 0;0 1], [0 1;1 0], [0 -im;im  0], [1 0;0 -1]
    pauli = Dict(0 => I, 1 => X, 2 => Y, 3 => Z)
    generator = 0.0 * kron(I, I)#pick custom phase   
    # Two-qubit σᵃ⊗σᵇ terms
    param_idx = 1
    for a in 1:3, b in 1:3
        term = kron(pauli[a], pauli[b])
        generator += -im * θ_vec[param_idx] * term
        param_idx += 1
    end
    #Single-qubit σᵃ⊗I terms
    for a in 1:3
        term = kron(pauli[a], pauli[0])
        generator += -im * θ_vec[param_idx] * term
        param_idx += 1
    end
    #Single-qubit I⊗σᵇ terms
    for b in 1:3
        term = kron(pauli[0], pauli[b]) 
        generator += -im * θ_vec[param_idx] * term
        param_idx += 1
    end
    return exp(generator)
end

function TwoQubitUnitary_ITensor_from_matrix(Umat::Matrix{ComplexF64}, s1::Index, s2::Index)
    @assert size(Umat) == (4, 4) "Matrix must be 4x4 for two qubits"

    s1p = prime(s1)
    s2p = prime(s2)

    U_tensor = ITensor(s1p, s2p, s1, s2)

    for b1 in 1:2, b2 in 1:2, a1 in 1:2, a2 in 1:2
        row = 2*(b1 - 1) + b2
        col = 2*(a1 - 1) + a2
        U_tensor[s1p => b1, s2p => b2, s1 => a1, s2 => a2] = Umat[row, col]
    end

    return U_tensor
end

function apply_ladder_circuit(θ_vec::Vector{Float64}, psi0::MPS)
    psi = copy(psi0)

    N = length(siteinds(psi))
    U2qubit = generate_2qubit_unitary(θ_vec)

    for i in 1:(N - 1)
        s1, s2 = siteind(psi, i), siteind(psi, i + 1)
        U = TwoQubitUnitary_ITensor_from_matrix(U2qubit, s1, s2)
        psi = apply(U, psi, [i, i+1]; maxdim=Inf)
    end

    psi = truncate(psi; maxdim=2)
    normalize!(psi)
    return psi
end

function optimize_fidelity( 
    Psi::MPS,  
    psi0::MPS,
    initial_params::Union{Nothing, Vector{Float64}}=nothing,
    max_iter=500
    )

    Random.seed!(123)
    if isnothing(initial_params)
        initial_params = 0.01 * rand(15)
    end

    function objective(p)
        psi = apply_ladder_circuit(p, psi0)
        cost = -abs2(inner(psi, Psi))
        return cost
    end
    
    result = optimize(objective, initial_params, BFGS(), 
                        Optim.Options(iterations=max_iter, show_trace=false))
    opt_params = Optim.minimizer(result)
    return opt_params
end

function generate_GHZ_state(psi0::MPS)
    params_GHZ = zeros(15)
    params_GHZ[7] = pi / 4
    params_GHZ[12] = - pi / 4
    params_GHZ[13] = - pi / 4
    psi = apply_ladder_circuit(params_GHZ, psi0)

    sites = siteinds(psi0)
    # Create |00...0> state
    state_0 = productMPS(sites, "Up")
    # Create |11...1> state
    state_1 = productMPS(sites, "Dn")
    # Create the superposition: (|00...0> + |11...1>)/sqrt(2)
    ghz_state = add(state_0, state_1)
    normalize!(ghz_state)
    fidelity_ghz = abs2(inner(psi, ghz_state))
    println("Fidelity wrt to GHZ = $(round(fidelity_ghz, digits=4))")

    return ghz_state, params_GHZ, :GHZ
end

function generate_cluster_state(psi0::MPS)
    sites = siteinds(psi0)

    # First create |+>⊗|+>...⊗|+> state
    cluster_state = MPS(sites, n -> "X+")
    # Apply CZ gates between neighboring qubits
    for j in 1:(length(sites) - 1)
        # Create CZ gate between sites j and j+1
        s1 = sites[j]
        s2 = sites[j + 1]
        
        # CZ = |00><00| + |01><01| + |10><10| - |11><11|
        # In computational basis, this corresponds to a phase flip when both qubits are in |1> state
        proj00 = op("ProjUp", s1) * op("ProjUp", s2)
        proj01 = op("ProjUp", s1) * op("ProjDn", s2)
        proj10 = op("ProjDn", s1) * op("ProjUp", s2)
        proj11 = op("ProjDn", s1) * op("ProjDn", s2)
        
        cz = proj00 + proj01 + proj10 - proj11
        
        # Apply the gate
        cluster_state = apply(cz, cluster_state, [j, j+1])
        normalize!(cluster_state)
    end

    params_cluster = optimize_fidelity(cluster_state, psi0)
    psi = apply_ladder_circuit(params_cluster, psi0)
    fidelity_cluster = abs2(inner(psi, cluster_state))
    println("Fidelity wrt to cluster state = $(round(fidelity_cluster, digits=4))")

    return cluster_state, params_cluster, :cluster
end