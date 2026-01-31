def _apply_flatquant(model, apply_flatquant_to_model, args, trainloader, logger):
    if not args.quantize:
        return model
    model = apply_flatquant_to_model(args, model)
    logger.info("Finished applying FlatQuant to model.")
    if args.act_sparsity:
        configure_act_sparsity(model, args, logger)
    if args.resume:
        flat_utils.load_flat_parameters(args, model)
    elif args.reload_matrix:
        flat_utils.load_flat_matrices(args, model, path=args.matrix_path)
    elif (args.cali_trans or args.add_diag or args.lwc or args.lac):
        train_utils.cali_flat_quant(args, model, trainloader, utils.DEV, logger=logger)
    if args.save_matrix and not args.reload_matrix:
        flat_utils.save_flat_matrices(args, model)
    flat_utils.reparameterize_model(model)
    logger.info("Finished reparameterize model.")
    return model