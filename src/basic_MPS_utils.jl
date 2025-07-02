# Define basic operators
function pauli_operators(s)
    sigmax = 2 * op("Sx", s)
    sigmay = 2 * op("Sy", s)
    sigmaz = 2 * op("Sz", s)
    id = op("Id", s)
    return [id, sigmax, sigmay, sigmaz]  # σ⁰, σˣ, σʸ, σᶻ
end

# Define the Hadamard gate
function hadamard(s)
    sx = op("Sx", s)
    sz = op("Sz", s)
    h = (sx + sz) * sqrt(2.0)
    return h
end