# two-qubit unitary, acting on sites i and i + 1
# full 15 parameter encoding
function create_15param_unitary(qr::QuantumResource1D, i::Int64, params::Union{Nothing, Vector{Float64}}=nothing)
    sites = qr.sites
    s1 = sites[i]
    s2 = sites[i+1]

    if isnothing(params)
        params = copy(qr.θ_vec)
    end

    if length(params) != 15
        @error "Provided circuit parameters don't encode 15 param 2-qubit unitary"
    end
    
    # Get Pauli operators for both sites
    ops1 = pauli_operators(s1)  # [I, σx, σy, σz] for site 1
    ops2 = pauli_operators(s2)  # [I, σx, σy, σz] for site 2
    
    # Initialize the generator of the unitary (Hermitian operator)
    generator = 0.0 * ops1[1] * ops2[1]#pick custom phase 
    
    # Two-qubit σᵃ⊗σᵇ terms
    param_idx = 1
    for a in 2:4, b in 2:4
        term = ops1[a] * ops2[b]
        generator += -im * params[param_idx] * term
        param_idx += 1
    end

    #Single-qubit σᵃ⊗I terms
    for a in 2:4
        term = ops1[a] * ops2[1]
        generator += -im * params[param_idx] * term
        param_idx += 1
    end

    #Single-qubit I⊗σᵇ terms
    for b in 2:4
        term = ops1[1] * ops2[b]
        generator += -im * params[param_idx] * term
        param_idx += 1
    end

    return exp(generator)
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
    for i in 1:(N-1)
        if circuit == :long_15_param_circuit
            U = create_15param_unitary(qr, i, params)
        end
        psi = apply(U, psi, [i, i+1]; maxdim=2)
    end

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

function generate_aux_ops(sites::IndexSet)
    N = length(sites)
    aux_ops = Vector{MPO}()

    X_string = ITensor(1)
    for j in 1:N
        s = sites[j]
        X_string *= 2 * op("Sx", s)
    end
    X_string_mpo = MPO(X_string, sites)
    push!(aux_ops, X_string_mpo)
    
    O_string = ITensor(1)
    O_string *= 2 * op("Sz", sites[1])
    O_string *= 2 * op("Sy", sites[2])
    for j in 3:(N - 2)
        s = sites[j]
        O_string *= 2 * op("Sx", s)
    end
    O_string *= 2 * op("Sy", sites[N - 1])
    O_string *= 2 * op("Sz", sites[N])
    O_string_mpo = MPO(O_string, sites)
    push!(aux_ops, (-1)^N * O_string_mpo)
    
    ZZ_term = AutoMPO()
    for j in 1:(N-1)
        ZZ_term .+= 4.0, "Sz", j, "Sz", j + 1
    end
    ZZ_term_mpo = MPO(ZZ_term, sites)
    push!(aux_ops, ZZ_term_mpo)
    
    return aux_ops
end

function cost_function(qr::QuantumResource1D, g::Float64, psi::Union{MPS, Nothing}=nothing)
    if isnothing(psi)
        psi = qr.psi
    end
    return - 0.5 * (1 - g) * real(inner(psi', qr.aux_ops[2], psi)) - 
           0.5 * (1 + g) * real(inner(psi', qr.aux_ops[3], psi)) / (qr.N - 1)
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
    result = optimize(objective, θ_in, BFGS(), Optim.Options(iterations=max_iter, show_trace=true))

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
    if abs(real(inner(psi', qr.aux_ops[2], psi))) > 1e-7 #O-string expectation value
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
