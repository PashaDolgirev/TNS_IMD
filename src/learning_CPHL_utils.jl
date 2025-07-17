function generate_CPHL_CorrMat(psi::MPS, op_list::Vector{MPO}, H0::MPO)
    n_ops = length(op_list)
    psi_list = Vector{MPS}(undef, n_ops)
    G_mat = Matrix{Float64}(undef, n_ops, n_ops)
    exp_vals = Vector{Float64}(undef, n_ops)
    v_vec = Vector{Float64}(undef, n_ops) #aux vector
    ψH0 = H0 * psi
    EH0 = real(inner(psi, ψH0))

    for j in 1:n_ops
        ψj = op_list[j] * psi
        psi_list[j] = ψj
        exp_vals[j] = real(inner(psi', op_list[j], psi))
    end

    for i in 1:n_ops, j in 1:n_ops
        G_mat[i, j] = real(inner(psi_list[i]', psi_list[j])) - exp_vals[i] * exp_vals[j]
    end

    for j in 1:n_ops
        v_vec[j] = EH0 * exp_vals[j] - real(inner(ψH0', psi_list[j]))
    end

    G_mat = 0.5 * (G_mat + G_mat')

    return G_mat, exp_vals, v_vec
end

function CollectAuxHL(cphl::CPHLSolver)
    for ind_g in 1:cphl.N_g
        G_g, e_g, v_g = generate_CPHL_CorrMat(cphl.ψ_opt_list[ind_g], cphl.op_list, cphl.H0_list[ind_g])
        cphl.GMat_list[ind_g] = G_g
        cphl.exp_val_list[ind_g] = e_g
        cphl.v_vec_list[ind_g] = v_g
    end
    # println("Successful collection of auxilary elements for HL")
end


function GenerateUpdCMat(cphl::CPHLSolver, α::Float64=1e-4)
    CollectAuxHL(cphl)

    custom_weights = 3.0 * [1.0; 1.0; 1.0; 1.0; 0.5;
                            2.0; 2.0; 
                            3.0; 2.5; 3.0; 3.0; 3.0; 3.0; 3.0; 3.0; 3.0]
    CMat_upd = zeros(Float64, cphl.N_g, cphl.N_op)
    for ind_g in 1:cphl.N_g
        G_g, e_g, v_g = copy(cphl.GMat_list[ind_g]), copy(cphl.exp_val_list[ind_g]), copy(cphl.v_vec_list[ind_g])

        c_upd = (G_g + Diagonal(custom_weights)) \ (v_g - α * e_g)
        CMat_upd[ind_g, :] .= c_upd
    end 
    return CMat_upd
end


function LearnHarmonics(cphl::CPHLSolver, delta_step::Float64=0.2, Max_iter::Int=20, tol::Float64=1e-2, β::Float64=0.2)
    λ_min = 0.01
    cphl.OStringWeight = copy(cphl.OrgWeight)
    for iter in 1:Max_iter
        C_mat_upd = GenerateUpdCMat(cphl)
        AlphaMat_upd = EstimateHarmFromCMat(C_mat_upd, cphl.HarmFromVecMat)
        current_error = norm( AlphaMat_upd - cphl.ALPHAMat)
        println("Current error = $(current_error); λ weight = $(cphl.OStringWeight)")

        cphl.ALPHAMat = (1 - delta_step) * cphl.ALPHAMat + delta_step * AlphaMat_upd
        cphl.CMat = GenerateCMatFromALPHAMat(cphl.ALPHAMat, cphl.DesignMat)

        if current_error < tol
            cphl.flag_convegence = true
            println("Converged in $(iter) iterations")
            break
        end
        if cphl.OStringWeight < λ_min
            cphl.OStringWeight = 0.0 
        else
            cphl.OStringWeight = copy(cphl.OrgWeight) / (1 + β * iter)
        end
        
        SetUpHamiltonians(cphl)
        OptimizeCPDMRG(cphl)
    end
    cphl.OrgWeight = copy(cphl.OStringWeight)
    println("Updated coefficients")
end