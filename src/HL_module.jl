module HL1D

include("basic_MPS_utils.jl")

using ITensors, ITensorMPS, Optim, LinearAlgebra, Printf, Random

export HamiltonianLearner, generate_Hamiltonianm, generate_correlation_matrix, LearnHamiltonian_corr_mat

mutable struct HamiltonianLearner
    N::Int
    sites::IndexSet
    psi::MPS                    # state vector
    coeff_list::Vector{Float64} # learned coefficients
    op_list::Vector{AutoMPO}    # Hamiltonian ansatz
    range::Int                  # encodes locality of the Hamiltonian

    method::Symbol              # :corr_mat or :IQA
    G_mat::Matrix{Float64}      # correlation matrix

    H::MPO                      # learned Hamiltonian
    ψ_GS::MPS                   # ground state

    E_GS::Float64               # ground state energy
    E_psi::Float64              # energy of psi
    Fidelity::Float64     
    λ_min::Float64      
end

# generates translationally invariant Hamiltonian terms
# we consider one, two, and three body Pauli terms 
# bulk of each generated term commutes with the string X1*X2*...*XN
function generate_custom_op_list(sites::IndexSet, range::Int)
    N = length(sites)

    op_list = AutoMPO[]

    #single Pauli term
    X_term = AutoMPO()
    for j in 1:N
        X_term .+= 2.0, "Sx", j
    end
    push!(op_list, X_term)

    #two-body Pauli terms
    for l in 1:range
        op_XX = AutoMPO()
        op_YY = AutoMPO()
        op_ZZ = AutoMPO()
        op_YZ = AutoMPO()
        op_ZY = AutoMPO()

        for j in 1:(N - l)
            op_XX .+= 4.0, "Sx", j, "Sx", j + l
            op_YY .+= 4.0, "Sy", j, "Sy", j + l
            op_ZZ .+= 4.0, "Sz", j, "Sz", j + l
            op_YZ .+= 4.0, "Sy", j, "Sz", j + l
            op_ZY .+= 4.0, "Sz", j, "Sy", j + l
        end

        push!(op_list, op_XX)
        push!(op_list, op_YY)
        push!(op_list, op_ZZ)
        push!(op_list, op_YZ)
        push!(op_list, op_ZY)
    end
    
    #three-body Pauli terms
    for l1 in 1:(range - 1)
        for l2 in (l1 + 1):range

            op_XXX = AutoMPO()

            op_YXY = AutoMPO()
            op_ZXZ = AutoMPO()
            op_YXZ = AutoMPO()
            op_ZXY = AutoMPO()

            op_YYX = AutoMPO()
            op_ZZX = AutoMPO()
            op_YZX = AutoMPO()
            op_ZYX = AutoMPO()

            op_XYY = AutoMPO()
            op_XZZ = AutoMPO()
            op_XYZ = AutoMPO()
            op_XZY = AutoMPO()

            for j in 1:(N - l2)
                op_XXX .+= 8.0, "Sx", j, "Sx", j + l1, "Sx", j + l2

                op_YXY .+= 8.0, "Sy", j, "Sx", j + l1, "Sy", j + l2
                op_ZXZ .+= 8.0, "Sz", j, "Sx", j + l1, "Sz", j + l2
                op_YXZ .+= 8.0, "Sy", j, "Sx", j + l1, "Sz", j + l2
                op_ZXY .+= 8.0, "Sz", j, "Sx", j + l1, "Sy", j + l2

                op_YYX .+= 8.0, "Sy", j, "Sy", j + l1, "Sx", j + l2
                op_ZZX .+= 8.0, "Sz", j, "Sz", j + l1, "Sx", j + l2
                op_YZX .+= 8.0, "Sy", j, "Sz", j + l1, "Sx", j + l2
                op_ZYX .+= 8.0, "Sz", j, "Sy", j + l1, "Sx", j + l2

                op_XYY .+= 8.0, "Sx", j, "Sy", j + l1, "Sy", j + l2
                op_XZZ .+= 8.0, "Sx", j, "Sz", j + l1, "Sz", j + l2
                op_XYZ .+= 8.0, "Sx", j, "Sy", j + l1, "Sz", j + l2
                op_XZY .+= 8.0, "Sx", j, "Sz", j + l1, "Sy", j + l2
            end
            push!(op_list, op_XXX)

            push!(op_list, op_YXY)
            push!(op_list, op_ZXZ)
            push!(op_list, op_YXZ)
            push!(op_list, op_ZXY)

            push!(op_list, op_YYX)
            push!(op_list, op_ZZX)
            push!(op_list, op_YZX)
            push!(op_list, op_ZYX)

            push!(op_list, op_XYY)
            push!(op_list, op_XZZ)
            push!(op_list, op_XYZ)
            push!(op_list, op_XZY)
        end
    end
    return op_list
end

function HamiltonianLearner(psi::MPS, range::Int=2,
    method::Union{Symbol, Nothing}=nothing
    )

    sites = siteinds(psi)
    N = length(sites)
    if isnothing(method)
        method = :corr_mat
    end

    op_list = generate_custom_op_list(sites, range)
    coeff_list = zeros(Float64, length(op_list))

    G_mat = zeros(Float64, 0, 0)

    H = MPO()
    ψ_GS = MPS()    

    return HamiltonianLearner(N, sites, copy(psi), coeff_list, op_list, range, method, G_mat, H, ψ_GS, 0.0, 0.0, 0.0, 1.0)
end

include("HL1D_utils.jl")

end