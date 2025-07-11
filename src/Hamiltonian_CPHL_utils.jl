function generate_CPHL_op_list(sites::IndexSet)
    N = length(sites)

    op_list = MPO[]

    #two-body Pauli terms
    op_XX_nn = AutoMPO()
    op_YY_nn = AutoMPO()
    op_XX_nnn = AutoMPO()
    op_YY_nnn = AutoMPO()
    op_ZZ_nnn = AutoMPO()   

    for j in 1:(N - 1)
        op_XX_nn .+= 4.0, "Sx", j, "Sx", j + 1
        op_YY_nn .+= 4.0, "Sy", j, "Sy", j + 1
    end

    for j in 1:(N - 2)
        op_XX_nnn .+= 4.0, "Sx", j, "Sx", j + 2
        op_YY_nnn .+= 4.0, "Sy", j, "Sy", j + 2
        op_ZZ_nnn .+= 4.0, "Sz", j, "Sz", j + 2
    end

    push!(op_list, MPO(op_XX_nn, sites))
    push!(op_list, MPO(op_YY_nn, sites))
    push!(op_list, MPO(op_XX_nnn, sites))
    push!(op_list, MPO(op_YY_nnn, sites))
    push!(op_list, MPO(op_ZZ_nnn, sites))

    return op_list
end

function generate_CPHL_Hamiltonian(g::Float64, coeff_list::Vector{Float64}, sites::IndexSet)
    
    N = length(sites)
    H = AutoMPO()

    for j in 1:(N - 1)
        H .+= -2.0 * (1 + g), "Sz", j, "Sz", j + 1
    end

    for j in 1:(N - 2)
        H .+= -4.0 * (1 - g), "Sz", j, "Sx", j + 1, "Sz", j + 2
    end
    H .+= -2.0 * (1 - g), "Sx", 1, "Sz", 2
    H .+= -2.0 * (1 - g), "Sz", N - 1, "Sx", N

    ind_op = 1

    for j in 1:(N - 1)
        H .+= 4.0 * coeff_list[ind_op], "Sx", j, "Sx", j + 1
        H .+= 4.0 * coeff_list[ind_op + 1], "Sy", j, "Sy", j + 1
    end

    for j in 1:(N - 2)
        H .+= 4.0 * coeff_list[ind_op + 2], "Sx", j, "Sx", j + 2
        H .+= 4.0 * coeff_list[ind_op + 3], "Sy", j, "Sy", j + 2
        H .+= 4.0 * coeff_list[ind_op + 4], "Sz", j, "Sz", j + 2
    end
    ind_op += 5

    return MPO(H, sites)
end
