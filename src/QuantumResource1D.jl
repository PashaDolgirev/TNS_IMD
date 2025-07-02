module QR1D

include("basic_MPS_utils.jl")

using ITensors, ITensorMPS, Optim, LinearAlgebra, Printf, Random

export QuantumResource1D, create_15param_unitary, apply_ladder_circuit,
        AddQR, generate_GHZ_state,
        cost_function, optimize_circuit, optimize_circuit_QR

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

    aux_ops::Vector{MPO}            #auxilary operators used to generate the cost function
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

function QuantumResource1D(N, psi0::Union{MPS, Nothing}=nothing, method::Union{Symbol, Nothing}=nothing)
    sites = siteinds("S=1/2", N)

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

    aux_ops = generate_aux_ops(sites)

    return QuantumResource1D(N, sites, copy(psi0), copy(psi0), method, θ_vec, PsiQR, θQR, NamesQR, aux_ops)
end

include("QR1D_utils.jl")

end
