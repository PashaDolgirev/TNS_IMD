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
        H .+= 4.0 * coeff_list[1], "Sz", j, "Sz", j + 1
    end

    for j in 1:(N - 2)
        H .+= 8.0 * coeff_list[2], "Sz", j, "Sx", j + 1, "Sz", j + 2
    end
    H .+= 4.0 * coeff_list[2], "Sx", 1, "Sz", 2
    H .+= 4.0 * coeff_list[2], "Sz", N - 1, "Sx", N

    ind_op = 3


    #two-body Pauli terms
    for j in 1:(N - 1)
        H .+= 4.0 * coeff_list[ind_op], "Sx", j, "Sx", j + 1
        H .+= 4.0 * coeff_list[ind_op + 1], "Sy", j, "Sy", j + 1
    end
    ind_op += 2
    
    #three-body Pauli terms
    #bulk terms
    for j in 1:(N - 2)
        H .+= 8.0 * coeff_list[ind_op], "Sx", j, "Sx", j + 1, "Sx", j + 2
        H .+= 8.0 * coeff_list[ind_op + 1], "Sy", j, "Sx", j + 1, "Sy", j + 2
    end
    ind_op += 2

    # #boundary terms - single Pauli
    H .+= 2.0 * coeff_list[ind_op], "Sx", 1
    H .+= 2.0 * coeff_list[ind_op + 1], "Sx", N
    H .+= 2.0 * coeff_list[ind_op + 2], "Sy", 1
    H .+= 2.0 * coeff_list[ind_op + 3], "Sy", N
    H .+= 2.0 * coeff_list[ind_op + 4], "Sz", 1
    H .+= 2.0 * coeff_list[ind_op + 5], "Sz", N
    ind_op += 6

    #boundary terms - two Paulis
    H .+= 4.0 * coeff_list[ind_op], "Sx", 1, "Sx", 2
    H .+= 4.0 * coeff_list[ind_op + 1], "Sx", 1, "Sy", 2
    H .+= 4.0 * coeff_list[ind_op + 2], "Sx", 1, "Sz", 2
    H .+= 4.0 * coeff_list[ind_op + 3], "Sy", 1, "Sx", 2
    H .+= 4.0 * coeff_list[ind_op + 4], "Sy", 1, "Sy", 2
    H .+= 4.0 * coeff_list[ind_op + 5], "Sy", 1, "Sz", 2
    H .+= 4.0 * coeff_list[ind_op + 6], "Sz", 1, "Sx", 2
    H .+= 4.0 * coeff_list[ind_op + 7], "Sz", 1, "Sy", 2
    H .+= 4.0 * coeff_list[ind_op + 8], "Sz", 1, "Sz", 2
    ind_op += 9

    H .+= 4.0 * coeff_list[ind_op], "Sx", N - 1, "Sx", N
    H .+= 4.0 * coeff_list[ind_op + 1], "Sx", N - 1, "Sy", N
    H .+= 4.0 * coeff_list[ind_op + 2], "Sx", N - 1, "Sz", N
    H .+= 4.0 * coeff_list[ind_op + 3], "Sy", N - 1, "Sx", N
    H .+= 4.0 * coeff_list[ind_op + 4], "Sy", N - 1, "Sy", N
    H .+= 4.0 * coeff_list[ind_op + 5], "Sy", N - 1, "Sz", N
    H .+= 4.0 * coeff_list[ind_op + 6], "Sz", N - 1, "Sx", N
    H .+= 4.0 * coeff_list[ind_op + 7], "Sz", N - 1, "Sy", N
    H .+= 4.0 * coeff_list[ind_op + 8], "Sz", N - 1, "Sz", N
    
    return H
end
