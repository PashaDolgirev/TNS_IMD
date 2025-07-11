function ConstructDesignMatrix(
    g_vals::Vector{Float64},
    M_max::Int, 
    reg_harmonic_param::Float64=1e-12
    )

    DesignMat = zeros(Float64, length(g_vals), M_max)

    for j in 1:length(g_vals), m in 1:M_max
        DesignMat[j, m] = sin(pi * m * (g_vals[j] + 1) / 2)
    end
    I_mat = Matrix{Float64}(I, size(DesignMat, 2), size(DesignMat, 2))
    HarmFromVecMat = (DesignMat' * DesignMat + reg_harmonic_param * I_mat) \ DesignMat'

    return DesignMat, HarmFromVecMat
end

function EstimateHarmFromCMat(
    CMat::Matrix{Float64},
    HarmFromVecMat::Matrix{Float64}
    )
    return HarmFromVecMat * CMat
end

function GenerateCMatFromALPHAMat(
    ALPHAMat::Matrix{Float64}, 
    DesignMat::Matrix{Float64}
    )
    return DesignMat * ALPHAMat
end