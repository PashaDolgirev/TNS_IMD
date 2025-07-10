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

function ResetState(qr::QuantumResource1D)
    qr.psi = copy(qr.psi0)
end

function apply_ladder_circuit(qr::QuantumResource1D, 
    params::Union{Nothing, Vector{Float64}}=nothing, 
    psi::Union{MPS, Nothing}=nothing, 
    circuit::Union{Symbol, Nothing}=nothing
    )

    if isnothing(psi)
        psi = copy(qr.psi0)
    end

    if isnothing(params)
        params = copy(qr.θ_vec)
    end

    if isnothing(circuit)
        circuit = :long_15_param_circuit
    end

    N = length(qr.sites)
    U2qubit = generate_2qubit_unitary(params)

    for i in 1:(N-1)
        s1, s2 = siteind(psi, i), siteind(psi, i + 1)
        U = TwoQubitUnitary_ITensor_from_matrix(U2qubit, s1, s2)
        psi = apply(U, psi, [i, i+1]; maxdim=Inf)
    end

    psi = truncate(psi; maxdim=2)
    normalize!(psi)
    qr.psi = copy(psi)
    return psi
end

# Fidelity optimization
function optimize_fidelity(qr::QuantumResource1D, 
    Psi::MPS, 
    initial_params::Union{Vector{Float64}, Nothing}=nothing, 
    max_iter=1000, 
    circuit::Union{Symbol, Nothing}=nothing
    )
    Random.seed!(123)

    function objective(p)
        psi = apply_ladder_circuit(qr, p)
        cost = -abs2(inner(psi, Psi))
        return cost
    end
    
    if isnothing(initial_params) && isnothing(circuit)
        initial_params = 0.01 * rand(15)
    end

    result = optimize(objective, initial_params, BFGS(), 
                        Optim.Options(iterations=max_iter, show_trace=false))
    opt_params = Optim.minimizer(result)
    return opt_params
end

#Add a quantum resource
function AddQR(qr::QuantumResource1D, 
    Psi::MPS, 
    θ_vec::Union{Vector{Float64}, Nothing}=nothing, 
    nameQR::Union{Symbol, Nothing}=nothing
    )

    normalize!(Psi)
    push!(qr.PsiQR, copy(Psi))
    push!(qr.NamesQR, isnothing(nameQR) ? :notnamed : nameQR)

    if isnothing(θ_vec)
        θ_vec = optimize_fidelity(qr, Psi)
        println("Fidelity of the optimized state = $(abs2(inner(apply_ladder_circuit(qr, θ_vec), Psi)))")
    end
    push!(qr.θQR, copy(θ_vec))
end

function generate_GHZ_state(qr::QuantumResource1D)
    params_GHZ = zeros(15)
    params_GHZ[7] = pi / 4
    params_GHZ[12] = - pi / 4
    params_GHZ[13] = - pi / 4
    psi = apply_ladder_circuit(qr, params_GHZ)

    # Create |00...0> state
    state_0 = productMPS(qr.sites, "Up")
    # Create |11...1> state
    state_1 = productMPS(qr.sites, "Dn")
    # Create the superposition: (|00...0> + |11...1>)/sqrt(2)
    ghz_state = add(state_0, state_1)
    normalize!(ghz_state)
    fidelity_ghz = abs2(inner(psi, ghz_state))
    println("Fidelity wrt to GHZ = $(round(fidelity_ghz, digits=4))")

    return ghz_state, params_GHZ, :GHZ
end

function generate_cluster_state(qr::QuantumResource1D)
    # First create |+>⊗|+>...⊗|+> state
    cluster_state = MPS(qr.sites, n -> "X+")
    # Apply CZ gates between neighboring qubits
    for j in 1:(length(qr.sites)-1)
        # Create CZ gate between sites j and j+1
        s1 = qr.sites[j]
        s2 = qr.sites[j+1]
        
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

    params_cluster = optimize_fidelity(qr, cluster_state)
    psi = apply_ladder_circuit(qr, params_cluster)
    fidelity_cluster = abs2(inner(psi, cluster_state))
    println("Fidelity wrt to cluster state = $(round(fidelity_cluster, digits=4))")

    return cluster_state, params_cluster, :cluster
end

function aux_exp_vals(qr::QuantumResource1D, psi::MPS)
    # sites = siteinds(psi)  # returns the IndexSet of physical (site) indices
    # N = length(sites)      # total number of sites

    # psi_modified = copy(psi)
    # for j in 1:N
    #     psi_modified = apply(2 * op("Sx", siteind(psi, j)), psi_modified; site=j)
    # end
    # X_expval = inner(psi', psi_modified)

    # psi_modified = copy(psi)
    # psi_modified = apply(2 * op("Sz", siteind(psi, 1)), psi_modified; site=1)
    # psi_modified = apply(2 * op("Sy", siteind(psi, 2)), psi_modified; site=2)
    # for j in 3:N - 2
    #     psi_modified = apply(2 * op("Sx", siteind(psi, j)), psi_modified; site=j)
    # end
    # psi_modified = apply(2 * op("Sy", siteind(psi, N - 1)), psi_modified; site=N-1)
    # psi_modified = apply(2 * op("Sz", siteind(psi, N)), psi_modified; site=N)
    # O_expval = (-1)^N * inner(psi', psi_modified)

    # ZZ_term_ev = inner(psi', ZZ_term, psi)
    
    return real(inner(psi', qr.X_string_mpo, psi)), real(inner(psi', qr.O_string_mpo, psi)), real(inner(psi', qr.ZZ_term_mpo, psi))
end


function cost_function(qr::QuantumResource1D, g::Float64, psi::Union{MPS, Nothing}=nothing)
    if isnothing(psi)
        psi = qr.psi
    end

    _, O_expval, ZZ_term_ev = aux_exp_vals(qr, psi)
    return - 0.5 * (1 - g) * O_expval - 0.5 * (1 + g) * ZZ_term_ev / (qr.N - 1)
end


function optimize_circuit(qr::QuantumResource1D, g::Float64, θ_in::Union{Vector{Float64}, Nothing}=nothing; max_iter=1000)
    
    function objective(p)
        psi = apply_ladder_circuit(qr, p)
        return cost_function(qr, g, psi)
    end
    
    if isnothing(θ_in)
        θ_in = 0.01 * rand(15)
    end
    println("Optimizing for g = $g...")
    result = optimize(objective, θ_in, BFGS(), Optim.Options(iterations=max_iter, show_trace=false))

    opt_params = Optim.minimizer(result)
    return opt_params
end

function optimize_circuit_QR(qr::QuantumResource1D, g::Float64, params_prev=nothing)

    all_costs = Float64[]
    all_params = Vector{Vector{Float64}}() 

    for n in 1:length(qr.PsiQR)
        params_current = optimize_circuit(qr, g, qr.θQR[n])
        psi = apply_ladder_circuit(qr, params_current)
        push!(all_params, params_current)
        push!(all_costs, cost_function(qr, g, psi))
    end

    if !isnothing(params_prev)
        params_current = optimize_circuit(qr, g, params_prev)
        psi = apply_ladder_circuit(qr, params_current)
        push!(all_params, params_current)
        push!(all_costs, cost_function(qr, g, psi))
    end

    best_cost, min_index = findmin(all_costs)
    best_params = all_params[min_index]

    psi = apply_ladder_circuit(qr, best_params)
    _, O_expval, _ = aux_exp_vals(qr, psi)

    if abs(O_expval) > 1e-7 #O-string expectation value
        cluster_index = findfirst(==(:cluster), qr.NamesQR)
        if min_index != cluster_index && abs(all_costs[cluster_index] - best_cost) < 1e-7
            best_cost = all_costs[cluster_index]
            best_params = all_params[cluster_index]
        end
    else
        ghz_index = findfirst(==(:GHZ), qr.NamesQR)
        if min_index != ghz_index && abs(all_costs[ghz_index] - best_cost) < 1e-7 
            best_cost = all_costs[ghz_index]
            best_params = all_params[ghz_index]
        end
    end

    return best_params, best_cost
end
