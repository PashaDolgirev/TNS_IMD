# stands for: continuous phase Hamiltonian learning
module CPHL

using ITensors, ITensorMPS, Optim, LinearAlgebra, Printf, Random

export CPHLSolver, ConstructDesignMatrix, generate_CPHL_op_list,
        generate_CPHL_Hamiltonian, SetUpHamiltonians, OptimizeCPCircuits,
        OptimizeCPDMRG, generate_CPHL_CorrMat, CollectAuxHL,
        GenerateUpdCMat

include("basic_MPS_utils.jl")
include("basic_CPHL_utils.jl")
include("Hamiltonian_CPHL_utils.jl")
include("basic_CPHL_circuit_utils.jl")

mutable struct CPHLSolver
    N_sites::Int
    sites::IndexSet

    g_vals::Vector{Float64}                 # available g-points
    N_g::Int                                # num of g points
    M_max::Int                              # harmonic truncation

    DesignMat::Matrix{Float64}
    HarmFromVecMat::Matrix{Float64}         # auxilary mat that converts a function on g-points into harmonics

    op_list::Vector{MPO}                    # Hamiltonian ansatz terms
    N_op::Int                               # number of operators in the HL ansatz

    ALPHAMat::Matrix{Float64}               # matrix containing all the harmonics ALPHAMat_{m, a} = alpha_{m, a}
    CMat::Matrix{Float64}                   # matrix of coefficients CMat_{j, a} = c_a(g_j)

    OStringWeight::Float64                  # enters the circuit optimization cost function
    OrgWeight::Float64                      # introduced for convergence reasons

    OString_mpo::MPO
    XString_mpo::MPO
    ZZ_term_mpo::MPO

    # psi0::MPS                               # reference state
    # θ_GHZ::Vector{Float64}                  # circuit parameters for the GHZ state
    # θ_cluster::Vector{Float64}              # circuit parameters for the cluster state
    # ThetaMat::Matrix{Float64}               # 15xN_g-matrix of all quantum cirquit coefficients 

    Hamiltonians::Vector{MPO}               # list of all N_g Hamiltonians
    H0_list::Vector{MPO}                    # list of bare N_g Hamiltonians
    ψ_GS_list::Vector{MPS}                  # list of ground states
    ψ_opt_list::Vector{MPS}                 # optimized states
    # ψ_circuit_list::Vector{MPS}             # list of variational ground states

    #various observables of interest
    E_GS_vals::Vector{Float64}      
    OString_GS_vals::Vector{Float64}
    XString_GS_vals::Vector{Float64}
    ZZ_GS_vals::Vector{Float64}

    Fidelities_vals::Vector{Float64}  
    OString_opt_vals::Vector{Float64}
    XString_opt_vals::Vector{Float64}
    ZZ_opt_vals::Vector{Float64}

    # Cost_circuit_vals::Vector{Float64}
    # E_circuit_vals::Vector{Float64}
    # OString_circuit_vals::Vector{Float64}
    # XString_circuit_vals::Vector{Float64}
    # ZZ_circuit_vals::Vector{Float64}

    GMat_list::Vector{Matrix{Float64}}      # all N_g correlation matrices
    exp_val_list::Vector{Vector{Float64}}   # all corresponding N_g exp values of op_list
    v_vec_list::Vector{Vector{Float64}}     # aux vector

    flag_convegence::Bool 

end

function CPHLSolver(N_sites::Int, 
    g_vals::Vector{Float64}, 
    OStringWeight::Float64=0.0,
    M_max::Int=3)
    OrgWeight = OStringWeight
    Random.seed!(1234)

    sites = siteinds("S=1/2", N_sites)
    N_g = length(g_vals)

    DesignMat, HarmFromVecMat = ConstructDesignMatrix(g_vals, M_max)

    op_list = generate_CPHL_op_list(sites)
    N_op = length(op_list)

    ALPHAMat = zeros(Float64, M_max, N_op)
    CMat = zeros(Float64, N_g, N_op)

    OString_mpo = generate_O_string_mpo(sites)
    XString_mpo = generate_X_string_mpo(sites)
    ZZ_term_mpo = generate_ZZ_term_mpo(sites)

    # psi0 = productMPS(sites, "Up") #all in the up-state (|0>)
    # h = hadamard(sites[1])#act with Hadamard on the first site
    # psi0 = apply(h, psi0, [1])
    # psi0 = normalize(psi0)

    # _, params_GHZ, _ = generate_GHZ_state(psi0)
    # _, params_cluster, _ = generate_cluster_state(psi0)

    # ThetaMat = zeros(Float64, 15, N_g)

    # for (ind, g) in enumerate(g_vals)
    #     θ_vec = 0.5 * (1 + g) * params_GHZ + 0.5 * (1 - g) * params_cluster
    #     ThetaMat[:, ind] = θ_vec
    # end

    Hamiltonians = Vector{MPO}(undef, N_g)
    H0_list = Vector{MPO}(undef, N_g)

    for (ind_g, g) in enumerate(g_vals)
        H0_list[ind_g] = generate_CPHL_H0(g, sites)
    end

    ψ_GS_list = Vector{MPS}(undef, N_g)
    ψ_opt_list = Vector{MPS}(undef, N_g)
    # ψ_circuit_list = Vector{MPS}(undef, N_g)

    for ind_g in 1:N_g
        ψ_GS_list[ind_g] = randomMPS(sites, 2)
        ψ_opt_list[ind_g] = randomMPS(sites, 2)
    end

    E_GS_vals = zeros(Float64, N_g) 
    OString_GS_vals = zeros(Float64, N_g)
    XString_GS_vals = zeros(Float64, N_g)
    ZZ_GS_vals = zeros(Float64, N_g)
    # Cost_circuit_vals = zeros(Float64, N_g)
    # E_circuit_vals = zeros(Float64, N_g)
    # OString_circuit_vals = zeros(Float64, N_g)
    # XString_circuit_vals = zeros(Float64, N_g)
    # ZZ_circuit_vals = zeros(Float64, N_g)

    Fidelities_vals = zeros(Float64, N_g)   
    OString_opt_vals = zeros(Float64, N_g)
    XString_opt_vals = zeros(Float64, N_g)
    ZZ_opt_vals = zeros(Float64, N_g)

    GMat_list = Vector{Matrix{Float64}}(undef, N_g)
    exp_val_list = Vector{Vector{Float64}}(undef, N_g)
    v_vec_list = Vector{Vector{Float64}}(undef, N_g)

    flag_convegence = false

    return CPHLSolver(N_sites, 
                        sites,
                        g_vals,
                        N_g,
                        M_max,
                        DesignMat,
                        HarmFromVecMat,
                        op_list,
                        N_op,
                        ALPHAMat,
                        CMat,
                        OStringWeight,
                        OrgWeight,
                        OString_mpo,
                        XString_mpo,
                        ZZ_term_mpo,
                        # psi0,
                        # params_GHZ,
                        # params_cluster,
                        # ThetaMat,
                        Hamiltonians,
                        H0_list,
                        ψ_GS_list,
                        ψ_opt_list,
                        # ψ_circuit_list,
                        # Fidelities_vals,
                        E_GS_vals,
                        OString_GS_vals,
                        XString_GS_vals,
                        ZZ_GS_vals,
                        # Cost_circuit_vals,
                        # E_circuit_vals,
                        # OString_circuit_vals,
                        # XString_circuit_vals,
                        # ZZ_circuit_vals
                        Fidelities_vals,
                        OString_opt_vals,
                        XString_opt_vals,
                        ZZ_opt_vals,
                        GMat_list,
                        exp_val_list,
                        v_vec_list,
                        flag_convegence
                        )
end

include("optimization_CPHL_routines.jl")
include("learning_CPHL_utils.jl")

end