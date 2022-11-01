    # using DynamicHMC, LogDensityProblems, Zygote
    # begin
    #     P = TransformedLogDensity(parameter_transformations, emulator_sampling_problem)
    #     ∇P = ADgradient(:Zygote, P);

    #     unscaled_chain_X_emulated_hmc = []
    #     chain_nll_emulated_hmc = []
    #     for initial_sample in ProgressBar(seed_X)

    #         # initialization = (q = build_parameters_named_tuple(training.free_parameters, initial_sample),)
    #         initialization = (q = initial_sample,)

    #         # Finally, we sample from the posterior. `chain` holds the chain (positions and
    #         # diagnostic information), while the second returned value is the tuned sampler
    #         # which would allow continuation of sampling.
    #         results = mcmc_with_warmup(Random.GLOBAL_RNG, ∇P, chain_length_emulate; initialization);

    #         # We use the transformation to obtain the posterior from the chain.
    #         chain_X_emulated_hmc = transform.(t, results.chain); # vector of NamedTuples
    #         samples = hcat(collect.(chain_X_emulated_hmc)...)
    #         samples = inverse_normalize_transform(samples, normalization_transformation)
    #         for j in 1:size(samples, 2)
    #             push!(unscaled_chain_X_emulated_hmc, samples[:,j])
    #             push!(chain_nll_emulated_hmc, emulator_sampling_problem, samples[:, j])
    #         end
    #     end
    # end
