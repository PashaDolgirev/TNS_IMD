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
        end
    end

    println("Successful pass, generated Hamiltonians and GSs")

end