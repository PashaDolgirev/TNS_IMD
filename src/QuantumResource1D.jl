module QR1D

include("basic_MPS_utils.jl")

using ITensors, ITensorMPS, Optim, LinearAlgebra, Printf, Random

export QuantumResource1D, generate_2qubit_unitary, apply_ladder_circuit,
        AddQR, generate_GHZ_state, aux_exp_vals,
        cost_function, optimize_circuit, optimize_circuit_QR, optimize_fidelity

mutable struct QuantumResource1D
    N::Int64                        #number of qubits
    sites::IndexSet
    psi0::MPS                       #reference state
    psi::MPS                        # circuit state
    method::Symbol                  #:ladder, :feedforward, etc.

    θ_vec::Vector{Float64}          #parameters of the circuit

    PsiQR::Vector{MPS}              #available resource states
    θQR::Vector{Vector{Float64}}    #parameters of those states
    NamesQR::Vector{Symbol}         #names of available resource states: :GHZ, :cluster, :plus, etc

    X_string_mpo::MPO
    O_string_mpo::MPO
    ZZ_term_mpo::MPO                    
end

function QuantumResource1D(N, sites, X_string_mpo::MPO, O_string_mpo::MPO, ZZ_term_mpo::MPO, psi0::Union{MPS, Nothing}=nothing, method::Union{Symbol, Nothing}=nothing)
    # sites = siteinds("S=1/2", N)

    if isnothing(psi0)
        psi0 = productMPS(sites, "Up") #all in the up-state (|0>)
        h = hadamard(sites[1])#act with Hadamard on the first site
        psi0 = apply(h, psi0, [1])
        psi0 = normalize(psi0);
    end

    if isnothing(method)
        method = :ladder
    end

    θ_vec = zeros(Float64, 15)
    PsiQR = Vector{MPS}()
    θQR = Vector{Vector{Float64}}()
    NamesQR = Vector{Symbol}()

    return QuantumResource1D(N, sites, copy(psi0), copy(psi0), method, θ_vec, PsiQR, θQR, NamesQR, 
                        X_string_mpo, O_string_mpo, ZZ_term_mpo)
end

include("QR1D_utils.jl")

end
