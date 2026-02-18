import os
import argparse
from datetime import datetime
import logging
from termcolor import colored
import pprint


supported_models = [
            './modelzoo/meta-llama/Llama-2-7b-hf',
            './modelzoo/meta-llama/Llama-2-13b-hf',
            './modelzoo/meta-llama/Llama-2-70b-hf',
            './modelzoo/meta-llama/Meta-Llama-3-8B',
            './modelzoo/meta-llama/Meta-Llama-3-70B',
            './modelzoo/meta-llama/Meta-Llama-3-8B-Instruct',
            './modelzoo/meta-llama/Meta-Llama-3-70B-Instruct',
            './modelzoo/meta-llama/Llama-3.1-8B', 
            './modelzoo/meta-llama/Llama-3.1-70B', 
            './modelzoo/meta-llama/Llama-3.1-8B-Instruct', 
            './modelzoo/meta-llama/Llama-3.1-70B-Instruct', 
            './modelzoo/meta-llama/Llama-3.3-70B-Instruct', 
            './modelzoo/Qwen/Qwen2.5-7B-Instruct', 
            './modelzoo/Qwen/Qwen2.5-32B-Instruct', 
            'meta-llama/Llama-3.1-8B',
            '/gemini/code/checkpoints/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b'
            ]
supported_datasets = ['wikitext2', 'c4', 'pile']
supported_eigen_datasets =["wikitext2", "arc", "mathqa", "gsm8k", "mmlu", "openbookqa", "winogrande"]


def parser_gen():
    parser = argparse.ArgumentParser()

    # General Arguments
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf',
                        help='Model to load.', choices=supported_models)
    parser.add_argument('--seed', type=int, default=0, help='Random seed for HuggingFace and PyTorch.')
    parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace token for model access.')

    # Activation Quantization Arguments
    parser.add_argument('--a_bits', type=int, default=16,
                        help='''Number of bits for inputs of the linear layers.
                                This applies to all linear layers in the model, including down-projection and out-projection.''')
    parser.add_argument('--a_groupsize', type=int, default=-1, 
                        help='Groupsize for activation quantization. Note that this should be the same as w_groupsize.')
    parser.add_argument('--a_asym', action="store_true", default=False,
                        help='Use asymmetric activation quantization.')
    parser.add_argument('--act_sparsity', type=str, default=None,
                        help='Activation N:M sparsity in form N:M (e.g., 2:4).')
    parser.add_argument('--act_sparsity_location', type=str, default='pre_trans',
                        help='Where to apply activation sparsity (pre_trans, pre_quant, post_quant).')
    parser.add_argument('--target_modules', type=str, default=None,
                        help='Comma-separated module name patterns to skip sparsity.')
    parser.add_argument('--weight_scoring', action='store_true', default=False,
                        help='Use weight-based scoring for activation sparsity.')

    # Weight Quantization Arguments
    parser.add_argument('--w_bits', type=int, default=16, 
                        help='Number of bits for weights of the linear layers.')
    parser.add_argument('--w_groupsize', type=int, default=-1, 
                        help='Groupsize for weight quantization. Note that this should be the same as a_groupsize.')
    parser.add_argument('--w_asym', action="store_true", default=False,
                        help='Use asymmetric weight quantization.')
    parser.add_argument('--gptq', action="store_true", default=False,
                        help='Quantize the weights using GPTQ. If w_bits < 16 and this flag is not set, use RtN.')
    parser.add_argument('--gptq_mse', action="store_true", default=False,
                        help='''Use MSE search to find the optimal clipping threshold for weight quantization. 
                                NOTE: Do not activate while using LWC.''')
    parser.add_argument('--percdamp', type=float, default=.01,
                        help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--act_order', action="store_true", default=False,
                        help='Use act-order in GPTQ.')

    # FlatQuant calibration Arguments
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs.')
    parser.add_argument('--cali_dataset', type=str, default='wikitext2',
                        help='Calibration dataset for FlatQuant and GPTQ.', choices=supported_datasets)
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration data samples for FlatQuant and GPTQ.')
    parser.add_argument('--cali_bsz', type=int, default=4,
                        help='Batch size for FlatQuant. Default is 4.')
    parser.add_argument("--flat_lr", type=float, default=1e-5, 
                        help='Learning rate for learnable transformation.')
    parser.add_argument("--flat_lr_min_ratio", type=float, default=1e-3,
                        help="Cosine LR minimum as a ratio of flat_lr (eta_min = flat_lr * ratio).")
    parser.add_argument("--flat_lr_tmax_mult", type=float, default=1.0,
                        help="Multiplier for cosine T_max to slow LR decay (>1 slows).")
    parser.add_argument("--use_stage2", action="store_true", default=False,
                        help="Enable stage-2: freeze perm logits and drop align loss.")
    parser.add_argument("--use_stage3", action="store_true", default=False,
                        help="Enable stage-3: freeze right matrices/perm logits and train only L1.")
    parser.add_argument("--stage2_start", type=int, default=None,
                        help="Epoch index to enter stage-2 (freeze perm logits).")
    parser.add_argument("--stage3_start", type=int, default=None,
                        help="Epoch index to enter stage-3 (freeze right; only L1).")
    parser.add_argument("--stage3_lr", type=float, default=None,
                        help="If set, override optimizer LR when entering stage-3 (scheduler kept).")
    parser.add_argument("--cali_trans", default=False, action="store_true", 
                        help="Enable calibration of transformations.")
    parser.add_argument("--add_diag", default=False, action="store_true", 
                        help="Add per-channel scaling.")
    parser.add_argument("--lwc", default=False, action="store_true", 
                        help="Use learnable weight clipping.")
    parser.add_argument("--lac", default=False, action="store_true", 
                        help="Use learnable activation clipping.")
    parser.add_argument('--resume', action="store_true", default=False, 
                        help='Resume from a previous checkpoint for evaluation.')
    parser.add_argument('--save_matrix', action="store_true", default=False, 
                        help='Save the matrix-style parameters of FlatQuant.')
    parser.add_argument('--reload_matrix', action="store_true", default=False, 
                        help='Reload matrices and the inverse matrices for evaluation.')
    parser.add_argument('--matrix_path', type=str, default=None,
                        help='Path to the pre-trained matrix-style parameters of FlatQuant.')
    parser.add_argument("--diag_init", type=str, default="sq_style", choices=["sq_style", "one_style"], 
                        help='The way to initialize per-channel scaling. Default is SmoothQuant style.')
    parser.add_argument("--diag_alpha", type=float, default=0.3, 
                        help='Hyperparameter for the SmoothQuant style initialization of per-channel scaling.')
    parser.add_argument("--warmup", default=False, action="store_true", help="Warm up the learning rate during training.")
    parser.add_argument("--deactive_amp", default=False, action="store_true", help="Disable AMP training.")
    parser.add_argument("--direct_inv", default=False, action="store_true", 
                        help="Use the inverse method in PyTorch to directly get the inverse matrix rather than SVD.")
    parser.add_argument("--separate_vtrans", default=False, action="store_true", 
                        help="Disable the integration of the vtrans transformation.")
    parser.add_argument("--dim_right", type=int, default=None,
                        help="Fix the right dimension for decomposition; "
                             "left dim is inferred as in_features / dim_right.")
    parser.add_argument("--dim2_matrix_path", type=str, default=None,
                        help="Path to dim_right=2 flat_matrices.pth for right-matrix alignment.")
    parser.add_argument("--dim2_loss_weight", type=float, default=0.0,
                        help="Weight for aligning dim_right=4 right-matrix main block to dim_right=2.")
    parser.add_argument("--soft_perm", action="store_true", default=False,
                        help="Use learnable soft permutation for dim_right alignment (Sinkhorn).")
    parser.add_argument("--soft_perm_reg", type=float, default=0.0,
                        help="Regularization weight to encourage near-permutation.")
    parser.add_argument("--use_x_perm", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable applying learned activation permutation (x_perm) during forward/eval.")
    parser.add_argument("--soft_x_perm", action="store_true", default=False,
                        help="Enable block-wise Sinkhorn permutation on activations (post AXB).")
    parser.add_argument("--use_x_perm_predictor", action=argparse.BooleanOptionalAction, default=False,
                        help="Predict per-token block-wise permutation logits instead of using shared logits.")
    parser.add_argument("--x_perm_num_clusters", type=int, default=4,
                        help="Number of shared permutation clusters for the x_perm predictor.")
    parser.add_argument("--x_perm_pred_hidden", type=int, default=128,
                        help="Hidden size of the x_perm predictor MLP.")
    parser.add_argument("--use_x_mask", action=argparse.BooleanOptionalAction, default=False,
                        help="Zero out channels 2:4 of each 4-wide block after x_perm.")
    parser.add_argument("--x_mask_mode", type=str, default="hard_fixed",
                        choices=["hard_fixed", "hard_top2", "soft_top2",
                                 "switch_top2_soft", "switch_top2_hard",
                                 "switch_top2_hard_ste"],
                        help="Masking mode after x_perm: hard_fixed zeros channels 2:4; "
                             "hard_top2 keeps top-2 magnitudes per group; soft_top2 uses 2*softmax gate; "
                             "switch_top2 chooses between dense and top2 per group; "
                             "switch_top2_hard_ste uses hard r with straight-through gradients.")
    parser.add_argument("--x_mask_tau", type=float, default=1.0,
                        help="Temperature for soft_top2 x_mask_mode (lower -> sharper).")
    parser.add_argument("--x_mask_r_thr", type=float, default=None,
                        help="Optional threshold for r in switch_top2 modes during eval; "
                             "if set, groups with r < thr use a hard mask.")
    parser.add_argument("--x_mask_r_mode", type=str, default="top2",
                        choices=["top2", "gate_raw"],
                        help="Hard mask source for r < thr in switch_top2 modes: "
                             "top2 uses mixed activations; gate_raw uses the raw gate.")
    parser.add_argument("--x_mask_gate_cost", type=float, default=0.0,
                        help="Weight for switch_top2 gate mean target loss (or L1 if target not set).")
    parser.add_argument("--x_mask_gate_target", type=float, default=None,
                        help="Target mean for switch_top2 gate (if set, use (mean-target)^2).")
    parser.add_argument("--x_mask_gate_entropy", type=float, default=0.0,
                        help="Weight for low-entropy regularization on switch_top2 gate (encourage binary).")
    parser.add_argument("--x_mask_energy_weight", type=float, default=0.0,
                        help="Weight for energy concentration loss on per-group 4d activations.")
    parser.add_argument("--x_mask_2hot_weight", type=float, default=0.0,
                        help="Weight for encouraging 2-of-4 (two-hot) gates in soft_top2 mode.")
    parser.add_argument("--x_mask_track_err", action="store_true", default=False,
                        help="Track per-group x_mask reconstruction error for key-group selection.")
    parser.add_argument("--x_mask_use_err", action="store_true", default=False,
                        help="Load x_mask_err_by_layer.pt and use saved key/non-key indices.")
    parser.add_argument("--x_mask_use_non_key", action="store_true", default=False,
                        help="Use non-key indices (instead of key indices) when building r.")
    parser.add_argument("--x_mask_key_ratio", type=float, default=None,
                        help="If set, mark top ratio of groups by error as key groups (per trans).")
    parser.add_argument("--x_mask_key_k", type=int, default=None,
                        help="If set, mark top-k groups by error as key groups (per trans).")
    parser.add_argument("--use_perm", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable soft permutation during reparameterization.")
    parser.add_argument("--use_comp_mask", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable complement-space masking during reparameterization.")
    parser.add_argument("--comp_zero_weight", type=float, default=0.0,
                        help="Weight for forcing complement-space outputs into quantization zero-zone.")
    parser.add_argument("--nm_zero_weight", type=float, default=0.0,
                        help="Weight for forcing nm-space outputs (reshape to hidden/4 x 4) into quantization zero-zone.")
    parser.add_argument("--comp_tau_alpha", type=float, default=1.0,
                        help="Scale factor for tau in complement-space loss (1.0 keeps original).")
    
    # KV-Cache Quantization Arguments
    parser.add_argument('--q_bits', type=int, default=16,
                        help='''Number of bits for queries quantization. 
                        Note that quantizing the queries needs another rotation for the keys/queries.''')
    parser.add_argument('--q_asym', action="store_true", default=False, 
                        help='Use asymmetric quantization for queries.')
    parser.add_argument('--q_groupsize', type=int, default=-1)

    parser.add_argument('--k_bits', type=int, default=16,
                        help='''Number of bits for K-cache quantization.
                        Note that quantizing the K-cache needs another rotation for the keys/queries.''')
    parser.add_argument('--k_asym', action="store_true", default=False, 
                        help='Use asymmetric quantization for K-cache.')
    parser.add_argument('--k_groupsize', type=int, default=-1, 
                    help='Groupsize for K-cache quantization.')

    parser.add_argument('--v_bits', type=int, default=16,
                        help='Number of bits for V-cache quantization.')
    parser.add_argument('--v_asym', action="store_true", default=False,
                        help='Use asymmetric quantization for V-cache.')
    parser.add_argument('--v_groupsize', type=int, default=-1)
    
    # Experiments Arguments
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory path.")
    parser.add_argument("--exp_name", type=str, default="exp", help="Experiment name.")

    # LM Eval Arguments
    parser.add_argument("--lm_eval", action="store_true", help="Evaluate the model on LM Eval tasks.")
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "lambada_openai"],
        help='Tasks to evaluate on LM Eval.')
    parser.add_argument('--lm_eval_batch_size', type=int, default=128, help='Batch size for evaluation with lm eval harness.')
    parser.add_argument('--num_fewshot', type=int, default=0, help='Number of few-shot examples for lm_eval.')
    parser.add_argument('--output_file', type=str, default=None, help='Optional path to save lm_eval results as JSON.')
    parser.add_argument(
        "--distribute_model",
        action="store_true",
        help="Distribute the model across multiple GPUs for evaluation.")

    # Add quantized_save flag
    parser.add_argument('--quantized_save', action = "store_true", default = False,
                        help = 'Save the quantized model checkpoint.')

    # Eigen compensation Arguments
    parser.add_argument('--eigen_compensation', action='store_true', default=False,
                        help='Apply eigen-based error compensation after weight quantization.')
    parser.add_argument('--eigen_dataset', type=str, default='wikitext2',
                        help='Dataset for eigen compensation calibration.', choices=supported_eigen_datasets)
    parser.add_argument('--eigen_nsamples', type=int, default=256,
                        help='Number of samples for eigen compensation.')
    parser.add_argument('--eigen_r', type=int, default=512,
                        help='Rank for eigen compensation.')

    # Observation / debug hooks
    parser.add_argument('--obs', action='store_true', default=False,
                        help='Enable activation observation and heatmap export during evaluation.')
    parser.add_argument('--obs_target', type=str, default='model.model.layers.0.mlp.up_proj',
                        help='Dot-path to the module whose input activations will be captured.')
    parser.add_argument('--obs_hook_position', type=str,
                        choices=['post_trans', 'pre_quant', 'post_quant', 'pre_wx'],
                        default='post_trans',
                        help='Hook point for activation capture: after trans, before quantization, after quantization, or before WX when not quantizing.')
    parser.add_argument('--obs_save_path', type=str, default=None,
                        help='Optional path to save the generated heatmap image. Defaults to <exp_dir>/obs_heatmap.png')

    args = parser.parse_args()
    if args.a_groupsize > -1:
        raise NotImplementedError
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.exp_name = f"{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.quantize = (args.w_bits < 16) or (args.a_bits < 16) or (args.q_bits < 16) or (args.k_bits < 16) or (args.v_bits < 16)
    # # cache path
    # args.cache_dir = os.path.join(args.output_dir, ".cache")
    # os.makedirs(args.cache_dir, exist_ok=True)
    # output path
    args.model_name = args.model.split("/")[-1]
    args.exp_dir = os.path.join(args.output_dir, args.model_name, f"w{args.w_bits}a{args.a_bits}", args.exp_name)
    os.makedirs(args.exp_dir, exist_ok=True)
    
    logger = create_logger(args.exp_dir)
    logger.info('Arguments: ')
    logger.info(pprint.pformat(vars(args)))
    logger.info('--' * 30)
    return args, logger


def create_logger(exp_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    log_file = os.path.join(exp_dir, f'log_rank{dist_rank}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger
