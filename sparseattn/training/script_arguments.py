from dataclasses import dataclass, field
from typing import Optional, List

from transformers import TrainingArguments as HfTrainingArguments


@dataclass
class ScriptArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_overrides_json: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "'{\"resid_pdrop\": 0.2}'"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    use_thinking: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the thinking mode (necessary to use this script "
                "with qwen models)."
            )
        },
    )
    should_log_loss: bool = field(
        default=False,
        metadata={"help": "Whether to log loss components during training"},
    )
    token_scaled_loss: bool = field(
        default=False,
        metadata={
            "help": "Whether to re-scale the loss by the number of valid training tokens instead of averaging loss across sequences and devices. This should be turned on for instruction tuning, especially when using synthetic data, as the valid training tokens vary across devices."
        },
    )

    tokenized_mds_train: List[str] = field(
        default_factory=list,
        metadata={"help": "Paths to tokenized training datasets in MDS format"},
    )
    tokenized_mds_validation: List[str] = field(
        default_factory=list,
        metadata={"help": "Paths to tokenized validation datasets in MDS format"},
    )
    tokenized_mds_test: List[str] = field(
        default_factory=list,
        metadata={"help": "Paths to tokenized test datasets in MDS format"},
    )


@dataclass
class TrainingArguments(HfTrainingArguments):
    min_lr_ratio: float = field(default=0.0)
    ordered: bool = field(default=False)
    cuda_empty_cache: bool = field(
        default=False, metadata={"help": "Empty cuda cache before every step."}
    )
    streaming_dataset: bool = field(
        default=True,
        metadata={
            "help": "Use streaming dataset, dataloader, and their ckpt and resume"
        },
    )
    seq_parallel_size: int = field(
        default=1,
        metadata={"help": "Sequence parallelism group size (1 is no parallelism)"},
    )

    # Arguments for prulong
    start_head_sparsity: float = field(
        default=0.0, metadata={"help": "The initial sparsity across attention heads."}
    )
    end_head_sparsity: float = field(
        default=0.95, metadata={"help": "The final sparsity across attention heads."}
    )
    mask_learning_rate: float = field(
        default=1e-3, metadata={"help": "The learning rate for the masks."}
    )
    reg_learning_rate: float = field(
        default=1e-3,
        metadata={"help": "The learning rate for the regularization lambdas."},
    )
    warmup_type: str = field(
        default="linear",
        metadata={"help": "The type of warmup schedule to use for the masks."},
    )
    sparsity_warmup_ratio: float = field(
        default=0.05,
        metadata={
            "help": "The fraction of training steps to warm the sparsity target over."
        },
    )
    disable_linear_regularization_term: bool = field(
        default=False,
        metadata={"help": "Whether to disable the linear regularization term."},
    )
    context_window_if_toggled: int = field(
        default=4096, metadata={"help": "The context window size to use when toggled."}
    )
    freeze_non_mask_parameters: bool = field(
        default=False, metadata={"help": "Whether to freeze the non-mask parameters."}
    )
    freeze_mask_parameters: bool = field(
        default=False, metadata={"help": "Whether to freeze the mask parameters."}
    )
    stripe_init_width_1: float = field(
        default=None,
        metadata={
            "help": "If initializing with a striped pattern, width_1 of the stripes. This parameter also determines if a striped pattern is used."
        },
    )
    stripe_init_width_2: float = field(
        default=None,
        metadata={
            "help": "If initializing with a striped pattern, width_2 of the stripes."
        },
    )
    stripe_init_start_with_keep: bool = field(
        default=False,
        metadata={
            "help": "If initializing with a striped pattern, whether to start with keep or drop."
        },
    )
    load_masks_from: str = field(
        default=None, metadata={"help": "Path to load masks from, if any."}
    )
    load_masks_sparsity: float = field(
        default=None, metadata={"help": "Sparsity to load masks with, if any."}
    )
    
    attention_type: str = field(
        default="None",
        metadata={
            "help": "The type of toggling to use. Currently supports: `streaming` and `local`."
        },
    )

    ## Streaming
    toggle_type: str = field(
        default="streaming",
        metadata={
            "help": "The type of toggling to use. Currently supports: `streaming` and `local`."
        },
    )
    sink_size: int = field(
        default=128,
        metadata={
            "help": "Number of sink tokens (will be rounded up to a multiple of 128)."
        },
    )

    topk_k: int = field(
        default=2048,
        metadata={"help": "The k value for top-k toggling."},
    )
    
    pooling_mode: str = field(
        default="first_token",
        metadata={
            "help": "The pooling mode to use. Options: 'first_token', 'mean'"
        },
    )
    
    enable_ada_sparsity: bool = field(
        default=False,
        metadata={"help": "Whether to enable layer-wise sparsity."},
    )

    # Layer-wise sparsity
    enable_layerwise_sparsity: bool = field(
        default=False,
        metadata={"help": "Whether to enable layer-wise sparsity."},
    )
    layerwise_sparsity_schedule: str = field(
        default="high-low-high",
        metadata={
            "help": "The schedule for layer-wise sparsity. Options: 'high-low-high', 'low-high-low'"
        },
    )
    layerwise_sparsity_min_ratio: float = field(
        default=0.5,
        metadata={
            "help": "The minimum ratio of layer-wise sparsity to global sparsity."
        },
    )
    layerwise_sparsity_max_ratio: float = field(
        default=1.0,
        metadata={
            "help": "The maximum ratio of layer-wise sparsity to global sparsity."
        },
    )
    layerwise_sparsity_power: float = field(
        default=1.0,
        metadata={
            "help": "The power to raise the layer index when computing layer-wise sparsity."
        },
    )
    layerwise_sparsity_weight: float = field(
        default=1.0,
        metadata={"help": "The weight of the layer-wise sparsity loss term."},
    )
    erank_analysis_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the effective rank analysis results file."},
    )
