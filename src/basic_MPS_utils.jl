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

function generate_X_string_mpo(sites)
    N = length(sites)
    W = Vector{ITensor}(undef, N)

    for i in 1:N
        s = sites[i]
        W[i] = 2 * op("Sx", s)
    end

    return MPO(W)
end

function generate_O_string_mpo(sites)
    N = length(sites)
    W = Vector{ITensor}(undef, N)

    W[1] =  2 * op("Sz", sites[1])
    W[2] =  2 * op("Sy", sites[2])
    for i in 3:(N - 2)
        s = sites[i]
        W[i] = 2 * op("Sx", s)
    end
    W[N - 1] =  2 * op("Sy", sites[N - 1])
    W[N] =  2 * op("Sz", sites[N])

    return MPO((-1)^N * W)
end

function generate_ZZ_term_mpo(sites)
    N = length(sites)
    ZZ_term = AutoMPO()
    for j in 1:(N - 1)
        ZZ_term .+= 4.0, "Sz", j, "Sz", j + 1
    end
    return MPO(ZZ_term, sites)
end
