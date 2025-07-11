function SetUpHamiltonians(cphl::CPHLSolver, verbose=true)

    maxdim_schedule = [10, 20, 100, 200, 400, 800, 800, 800]
    sweeps = Sweeps(length(maxdim_schedule))
    maxdim!(sweeps, maxdim_schedule...)
    cutoff!(sweeps, 1e-10)

    for (ind_g, g) in enumerate(cphl.g_vals)
        coeff_list_g = cphl.CMat[ind_g, :]
        H_g = generate_CPHL_Hamiltonian(g, coeff_list_g, cphl.sites)
        cphl.Hamiltonians[ind_g] = H_g

        θ_g = cphl.ThetaMat[:, ind_g]
        psi0_g = apply_ladder_circuit(θ_g, cphl.psi0)
        E_GS, ψ_GS = dmrg(H_g, psi0_g, sweeps)

        cphl.Ψ_GS_list[ind_g] = ψ_GS
        cphl.E_GS_vals[ind_g] = E_GS
        cphl.OString_GS_vals[ind_g] = real(inner(ψ_GS', cphl.OString_mpo, ψ_GS))
        cphl.XString_GS_vals[ind_g] = real(inner(ψ_GS', cphl.XString_mpo, ψ_GS))
        cphl.ZZ_GS_vals[ind_g] = real(inner(ψ_GS', cphl.ZZ_term_mpo, ψ_GS))

        if verbose
            println("g = $g, GS energy = $(E_GS)")
            flush(stdout)
        end
    end

    println("Successful pass, generated Hamiltonians and GSs")
end


function cost_function(psi::MPS, OStringWeight::Float64, H::MPO, OString_mpo::MPO)

    O_expval = real(inner(psi, OString_mpo, psi))
    Eng_psi = real(inner(psi, H, psi))

    return Eng_psi - OStringWeight * O_expval
end

function optimize_circuit(psi0::MPS, θ_in::Vector{Float64}, OStringWeight::Float64, H::MPO, OString_mpo::MPO; max_iter=1000)
    
    function objective(p)
        psi = apply_ladder_circuit(p, psi0)
        return cost_function(psi, OStringWeight, H, OString_mpo)
    end
    
    result = optimize(objective, θ_in, BFGS(), Optim.Options(iterations=max_iter, show_trace=false))

    opt_params = Optim.minimizer(result)
    return opt_params
end

function OptimizeCPCircuits(cphl::CPHLSolver, verbose=true)
    for (ind_g, g) in enumerate(cphl.g_vals)
        H_g = cphl.Hamiltonians[ind_g]
        θ_g = optimize_circuit(cphl.psi0, cphl.ThetaMat[:, ind_g], cphl.OStringWeight, H_g, cphl.OString_mpo)
        psi_g = apply_ladder_circuit(θ_g, cphl.psi0)

        cphl.Ψ_circuit_list[ind_g] = psi_g
        cphl.OString_circuit_vals[ind_g] = real(inner(psi_g', cphl.OString_mpo, psi_g))
        cphl.XString_circuit_vals[ind_g] = real(inner(psi_g', cphl.XString_mpo, psi_g))
        cphl.ZZ_circuit_vals[ind_g] = real(inner(psi_g', cphl.ZZ_term_mpo, psi_g))

        cphl.Fidelities_vals[ind_g] = abs2(inner(psi_g', cphl.Ψ_GS_list[ind_g]))
        cphl.E_circuit_vals[ind_g] = real(inner(psi_g', H_g, psi_g))
        cphl.Cost_circuit_vals[ind_g] = cost_function(psi_g, cphl.OStringWeight, H_g, cphl.OString_mpo)

        if verbose
            println("g = $g, variational energy = $(cphl.E_circuit_vals[ind_g])")
            flush(stdout)
        end
    end
    println("Successful pass of circuit optimization")
end





