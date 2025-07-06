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

    method::Symbol              # :corr_mat or :IQA
    G_mat::Matrix{Float64}      # correlation matrix
    exp_vals::Vector{Float64}   # expectation_vals wrt to psi

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
# Note that the boundary terms can explicitly break this symmetry
function generate_custom_op_list(sites::IndexSet)
    N = length(sites)

    op_list = AutoMPO[]

    #single Pauli term
    X_term = AutoMPO()
    for j in 1:N
        X_term .+= 2.0, "Sx", j
    end
    push!(op_list, X_term)

    #two-body Pauli terms
    op_XX = AutoMPO()
    op_YY = AutoMPO()
    op_ZZ = AutoMPO()
    op_YZ = AutoMPO()
    op_ZY = AutoMPO()

    for j in 1:(N - 1)
        op_XX .+= 4.0, "Sx", j, "Sx", j + 1
        op_YY .+= 4.0, "Sy", j, "Sy", j + 1
        op_ZZ .+= 4.0, "Sz", j, "Sz", j + 1
        op_YZ .+= 4.0, "Sy", j, "Sz", j + 1
        op_ZY .+= 4.0, "Sz", j, "Sy", j + 1
    end

    push!(op_list, op_XX)
    push!(op_list, op_YY)
    push!(op_list, op_ZZ)
    push!(op_list, op_YZ)
    push!(op_list, op_ZY)

    #three-body Pauli terms
    op_XXX = AutoMPO()
    op_YXY = AutoMPO()
    op_ZXZ = AutoMPO()
    op_YXZ = AutoMPO()
    op_ZXY = AutoMPO()

    #bulk terms
    for j in 1:(N - 2)
        op_XXX .+= 8.0, "Sx", j, "Sx", j + 1, "Sx", j + 2
        op_YXY .+= 8.0, "Sy", j, "Sx", j + 1, "Sy", j + 2
        op_ZXZ .+= 8.0, "Sz", j, "Sx", j + 1, "Sz", j + 2
        op_YXZ .+= 8.0, "Sy", j, "Sx", j + 1, "Sz", j + 2
        op_ZXY .+= 8.0, "Sz", j, "Sx", j + 1, "Sy", j + 2
    end
    #corresponding boundary terms
    # op_XXX .+= 4.0, "Sx", 1, "Sx", 2
    # op_XXX .+= 4.0, "Sx", N - 1, "Sx", N
    # op_YXY .+= 4.0, "Sx", 1, "Sy", 2
    # op_YXY .+= 4.0, "Sy", N - 1, "Sx", N
    # op_ZXZ .+= 4.0, "Sx", 1, "Sz", 2
    # op_ZXZ .+= 4.0, "Sz", N - 1, "Sx", N
    # op_YXZ .+= 4.0, "Sx", 1, "Sz", 2
    # op_YXZ .+= 4.0, "Sy", N - 1, "Sx", N
    # op_ZXY .+= 4.0, "Sx", 1, "Sy", 2
    # op_ZXY .+= 4.0, "Sz", N - 1, "Sx", N

    push!(op_list, op_XXX)
    push!(op_list, op_YXY)
    push!(op_list, op_ZXZ)
    push!(op_list, op_YXZ)
    push!(op_list, op_ZXY)

    #boundary terms - single Pauli
    X1 = AutoMPO(); X1 .+= 2.0, "Sx", 1
    XN = AutoMPO(); XN .+= 2.0, "Sx", N
    Y1 = AutoMPO(); Y1 .+= 2.0, "Sy", 1
    YN = AutoMPO(); YN .+= 2.0, "Sy", N
    Z1 = AutoMPO(); Z1 .+= 2.0, "Sz", 1
    ZN = AutoMPO(); ZN .+= 2.0, "Sz", N

    push!(op_list, X1)
    push!(op_list, XN)
    push!(op_list, Y1)
    push!(op_list, YN)
    push!(op_list, Z1)
    push!(op_list, ZN)

    #boundary terms - two Paulis
    XX1 = AutoMPO(); XX1 .+= 4.0, "Sx", 1, "Sx", 2
    XY1 = AutoMPO(); XY1 .+= 4.0, "Sx", 1, "Sy", 2
    XZ1 = AutoMPO(); XZ1 .+= 4.0, "Sx", 1, "Sz", 2
    YX1 = AutoMPO(); YX1 .+= 4.0, "Sy", 1, "Sx", 2
    YY1 = AutoMPO(); YY1 .+= 4.0, "Sy", 1, "Sy", 2
    YZ1 = AutoMPO(); YZ1 .+= 4.0, "Sy", 1, "Sz", 2
    ZX1 = AutoMPO(); ZX1 .+= 4.0, "Sz", 1, "Sx", 2
    ZY1 = AutoMPO(); ZY1 .+= 4.0, "Sz", 1, "Sy", 2
    ZZ1 = AutoMPO(); ZZ1 .+= 4.0, "Sz", 1, "Sz", 2
    push!(op_list, XX1)
    push!(op_list, XY1)
    push!(op_list, XZ1)
    push!(op_list, YX1)
    push!(op_list, YY1)
    push!(op_list, YZ1)
    push!(op_list, ZX1)
    push!(op_list, ZY1)
    push!(op_list, ZZ1)

    XXN = AutoMPO(); XXN .+= 4.0, "Sx", N - 1, "Sx", N
    XYN = AutoMPO(); XYN .+= 4.0, "Sx", N - 1, "Sy", N
    XZN = AutoMPO(); XZN .+= 4.0, "Sx", N - 1, "Sz", N
    YXN = AutoMPO(); YXN .+= 4.0, "Sy", N - 1, "Sx", N
    YYN = AutoMPO(); YYN .+= 4.0, "Sy", N - 1, "Sy", N
    YZN = AutoMPO(); YZN .+= 4.0, "Sy", N - 1, "Sz", N
    ZXN = AutoMPO(); ZXN .+= 4.0, "Sz", N - 1, "Sx", N
    ZYN = AutoMPO(); ZYN .+= 4.0, "Sz", N - 1, "Sy", N
    ZZN = AutoMPO(); ZZN .+= 4.0, "Sz", N - 1, "Sz", N
    push!(op_list, XXN)
    push!(op_list, XYN)
    push!(op_list, XZN)
    push!(op_list, YXN)
    push!(op_list, YYN)
    push!(op_list, YZN)
    push!(op_list, ZXN)
    push!(op_list, ZYN)
    push!(op_list, ZZN)
    
    return op_list
end

function HamiltonianLearner(psi::MPS, method::Union{Symbol, Nothing}=nothing)

    sites = siteinds(psi)
    N = length(sites)
    if isnothing(method)
        method = :corr_mat
    end

    op_list = generate_custom_op_list(sites)
    coeff_list = zeros(Float64, length(op_list))

    G_mat = zeros(Float64, 0, 0)
    exp_vals = zeros(Float64, 0)

    H = MPO()
    ψ_GS = MPS()    

    return HamiltonianLearner(N, sites, copy(psi), coeff_list, op_list, method, G_mat, exp_vals, H, ψ_GS, 0.0, 0.0, 0.0, 1.0)
end

include("HL1D_utils.jl")

end