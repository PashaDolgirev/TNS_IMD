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

    #three-body Pauli terms
    op_XXX = AutoMPO()
    op_YXY = AutoMPO()
    for j in 1:(N - 2)
        op_XXX .+= 8.0, "Sx", j, "Sx", j + 1, "Sx", j + 2
        op_YXY .+= 8.0, "Sy", j, "Sx", j + 1, "Sy", j + 2
    end

    push!(op_list, MPO(op_XXX, sites))
    push!(op_list, MPO(op_YXY, sites))

    #four Pauli terms
    op_YZZY = AutoMPO()
    op_ZYYZ = AutoMPO()
    for j in 1:(N - 3)
        op_YZZY .+= 16.0, "Sy", j, "Sz", j + 1, "Sz", j + 2, "Sy", j + 3
        op_ZYYZ .+= 16.0, "Sz", j, "Sy", j + 1, "Sy", j + 2, "Sz", j + 3
    end
    push!(op_list, MPO(op_YZZY, sites))
    push!(op_list, MPO(op_ZYYZ, sites))

    op_YXXY = AutoMPO()
    op_YYYY = AutoMPO()

    op_ZXXZ = AutoMPO()
    op_ZZZZ = AutoMPO()

    op_XXXX = AutoMPO()
    op_XYYX = AutoMPO()
    op_XZZX = AutoMPO()
    #four Pauli terms
    for j in 1:(N - 3)
        op_YXXY .+= 16.0, "Sy", j, "Sx", j + 1, "Sx", j + 2, "Sy", j + 3
        op_YYYY .+= 16.0, "Sy", j, "Sy", j + 1, "Sy", j + 2, "Sy", j + 3
        op_ZXXZ .+= 16.0, "Sz", j, "Sx", j + 1, "Sx", j + 2, "Sz", j + 3
        op_ZZZZ .+= 16.0, "Sz", j, "Sz", j + 1, "Sz", j + 2, "Sz", j + 3

        op_XXXX .+= 16.0, "Sx", j, "Sx", j + 1, "Sx", j + 2, "Sx", j + 3
        op_XYYX .+= 16.0, "Sx", j, "Sy", j + 1, "Sy", j + 2, "Sx", j + 3
        op_XZZX .+= 16.0, "Sx", j, "Sz", j + 1, "Sz", j + 2, "Sx", j + 3
    end
    push!(op_list, MPO(op_YXXY, sites))
    push!(op_list, MPO(op_YYYY, sites))
    push!(op_list, MPO(op_ZXXZ, sites))
    push!(op_list, MPO(op_ZZZZ, sites))
    push!(op_list, MPO(op_XXXX, sites))
    push!(op_list, MPO(op_XYYX, sites))
    push!(op_list, MPO(op_XZZX, sites))

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

    #three Pauli terms
    for j in 1:(N - 2)
        H .+= 8.0 * coeff_list[ind_op], "Sx", j, "Sx", j + 1, "Sx", j + 2
        H .+= 8.0 * coeff_list[ind_op + 1], "Sy", j, "Sx", j + 1, "Sy", j + 2
    end
    ind_op += 2

    #four Pauli terms
    for j in 1:(N - 3)
        H .+= 16.0 * coeff_list[ind_op], "Sy", j, "Sz", j + 1, "Sz", j + 2, "Sy", j + 3
        H .+= 16.0 * coeff_list[ind_op + 1], "Sz", j, "Sy", j + 1, "Sy", j + 2, "Sz", j + 3

        H .+= 16.0 * coeff_list[ind_op + 2], "Sy", j, "Sx", j + 1, "Sx", j + 2, "Sy", j + 3
        H .+= 16.0 * coeff_list[ind_op + 3], "Sy", j, "Sy", j + 1, "Sy", j + 2, "Sy", j + 3
        H .+= 16.0 * coeff_list[ind_op + 4], "Sz", j, "Sx", j + 1, "Sx", j + 2, "Sz", j + 3
        H .+= 16.0 * coeff_list[ind_op + 5], "Sz", j, "Sz", j + 1, "Sz", j + 2, "Sz", j + 3

        H .+= 16.0 * coeff_list[ind_op + 6], "Sx", j, "Sx", j + 1, "Sx", j + 2, "Sx", j + 3
        H .+= 16.0 * coeff_list[ind_op + 7], "Sx", j, "Sy", j + 1, "Sy", j + 2, "Sx", j + 3
        H .+= 16.0 * coeff_list[ind_op + 8], "Sx", j, "Sz", j + 1, "Sz", j + 2, "Sx", j + 3
        
    end
    ind_op += 9

    return MPO(H, sites)
end

function generate_CPHL_H0(g::Float64, sites::IndexSet)
    
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

    return MPO(H, sites)
end
