import argparse
import yaml
import ast
import os

import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser(description="evaluation on downstream tasks")
    parser.add_argument("--config", type=str, default=None, help="path to config file")
    parser.add_argument(
        "--tag", type=str, default="eval", help="tag to add to the output file"
    )

    # model setting
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="path to the model or model name",
    )
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument(
        "--use_vllm", action="store_true", help="whether to use vllm engine"
    )

    # data settings
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="comma separated list of dataset names",
    )
    parser.add_argument(
        "--demo_files",
        type=str,
        default=None,
        help="comma separated list of demo files",
    )
    parser.add_argument(
        "--test_files",
        type=str,
        default=None,
        help="comma separated list of test files",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="path to save the predictions"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="whether to the saved file"
    )
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument(
        "--num_workers", type=int, default=4, help="number of workers for data loading"
    )

    # dataset specific settings
    parser.add_argument(
        "--popularity_threshold",
        type=int,
        default=3,
        help="popularity threshold for popqa, in log scale",
    )

    # evaluation settings
    parser.add_argument(
        "--shots", type=int, default=2, help="total number of ICL demos"
    )
    parser.add_argument(
        "--input_max_length",
        type=str,
        default="8192",
        help="the maximum number of tokens of the input, we truncate the end of the context; can be separated by comma to match the specified datasets",
    )

    # generation settings
    parser.add_argument(
        "--do_sample",
        type=ast.literal_eval,
        choices=[True, False],
        default=False,
        help="whether to use sampling (false is greedy), overwrites temperature",
    )
    parser.add_argument(
        "--generation_max_length",
        type=str,
        default="10",
        help="max number of tokens to generate, can be separated by comma to match the specified datasets",
    )
    parser.add_argument(
        "--generation_min_length",
        type=int,
        default=0,
        help="min number of tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="generation temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0, help="top-p parameter for nucleus sampling"
    )
    parser.add_argument(
        "--stop_newline",
        type=ast.literal_eval,
        choices=[True, False],
        default=False,
        help="whether to stop generation at newline",
    )

    # model specific settings
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no_cuda", action="store_true", help="disable cuda")
    parser.add_argument(
        "--no_bf16", action="store_true", help="disable bf16 and use fp32"
    )
    parser.add_argument(
        "--no_torch_compile", action="store_true", help="disable torchcompile"
    )
    parser.add_argument(
        "--use_chat_template",
        type=ast.literal_eval,
        choices=[True, False],
        default=False,
        help="whether to use chat template",
    )
    parser.add_argument(
        "--rope_theta", type=int, default=None, help="override rope theta"
    )

    # misc
    parser.add_argument("--debug", action="store_true", help="for debugging")
    parser.add_argument(
        "--count_tokens",
        action="store_true",
        help="instead of running generation, just count the number of tokens (only for HF models not API)",
    )

    # sparseattn
    parser.add_argument("--sparseattn", action="store_true", help="e2e eval")

    # duoattn
    parser.add_argument(
        "--duoattn", type=str, default=None, help="path to the duoattn pattern"
    )
    parser.add_argument(
        "--duoattn_sparsity",
        type=float,
        default=None,
        help="sparsity of the duoattn pattern",
    )
    parser.add_argument(
        "--duoattn_sink", type=int, default=128, help="sink size of the duoattn pattern"
    )
    parser.add_argument(
        "--duoattn_sliding",
        type=int,
        default=1024,
        help="sliding size of the duoattn pattern",
    )
    parser.add_argument(
        "--duoattn_chunk_prefilling",
        type=int,
        default=None,
        help="use chunk prefilling",
    )
    parser.add_argument(
        "--duoattn_flipping",
        action="store_true",
        help="whether to flip the duoattn pattern",
    )

    # fastprefill
    parser.add_argument(
        "--fastprefill_threshold",
        type=float,
        default=0.95,
        help="threshold for fastprefill",
    )
    parser.add_argument(
        "--fastprefill_print_detail",
        type=bool,
        default=False,
        help="whether to print detail for fastprefill",
    )
    parser.add_argument(
        "--fastprefill_stride", type=int, default=16, help="stride for fastprefill"
    )
    parser.add_argument(
        "--fastprefill_metric", type=str, default=None, help="metric for fastprefill"
    )

    # minference, snapkv and pyramidkv
    parser.add_argument(
        "--minference",
        type=str,
        default=None,
        help="which method to use with minference",
    )
    parser.add_argument(
        "--minference_model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="the model name to use with minference",
    )
    parser.add_argument(
        "--minference_chunk_prefilling",
        type=int,
        default=None,
        help="use chunk prefilling",
    )
    parser.add_argument(
        "--minference_sparsity",
        type=float,
        default=None,
        help="token sparsity - overrides max_capacity_prompt",
    )
    parser.add_argument(
        "--minference_window_size",
        type=int,
        default=32,
        help="window size for minference",
    )
    parser.add_argument(
        "--minference_max_capacity_prompt",
        type=int,
        default=4096,
        help="max capacity prompt for minference",
    )
    parser.add_argument(
        "--minference_chunking_patch",
        action="store_true",
        help="patch in the last `k` tokens for all chunk",
    )
    parser.add_argument(
        "--minference_grouped_eviction",
        action="store_true",
        help="grouped eviction for pyramid/snapkv",
    )
    parser.add_argument(
        "--minference_compress_group_kvs",
        action="store_true",
        help="compress group kvs for pyramid/snapkv",
    )

    # locret
    parser.add_argument(
        "--locret_bin_file", type=str, default=None, help="path to the locret bin file"
    )
    parser.add_argument(
        "--locret_sparsity", type=float, default=None, help="KV sparsity for locret"
    )
    parser.add_argument(
        "--locret_budget_size",
        type=int,
        default=None,
        help="budget size for locret; overriden by locret_sparsity",
    )
    parser.add_argument(
        "--locret_local_len", type=int, default=100, help="local length for locret"
    )
    parser.add_argument(
        "--locret_stabilizers", type=int, default=2500, help="stabilizers for locret"
    )
    parser.add_argument(
        "--locret_chunk_prefilling",
        type=int,
        default=None,
        help="chunk size for locret",
    )

    args = parser.parse_args()
    config = yaml.safe_load(open(args.config)) if args.config is not None else {}
    parser.set_defaults(**config)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"output/{os.path.basename(args.model_name_or_path)}"

    if args.rope_theta is not None:
        args.output_dir = args.output_dir + f"-override-rope{args.rope_theta}"

    if not args.do_sample and args.temperature != 0.0:
        args.temperature = 0.0
        logger.info("overwriting temperature to 0.0 since do_sample is False")

    return args
/data1/anaconda3/envs/qqt/lib/python3.11/site-packages/tensor_parallel/imports.py:1: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
/data1/anaconda3/envs/qqt/lib/python3.11/site-packages/tensor_parallel/imports.py:1: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
/data1/anaconda3/envs/qqt/lib/python3.11/site-packages/tensor_parallel/imports.py:1: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
/data1/anaconda3/envs/qqt/lib/python3.11/site-packages/tensor_parallel/imports.py:1: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
/data1/anaconda3/envs/qqt/lib/python3.11/site-packages/tensor_parallel/imports.py:1: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
/data1/anaconda3/envs/qqt/lib/python3.11/site-packages/tensor_parallel/imports.py:1: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
/data1/anaconda3/envs/qqt/lib/python3.11/site-packages/tensor_parallel/imports.py:1: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
/data1/anaconda3/envs/qqt/lib/python3.11/site-packages/tensor_parallel/imports.py:1: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
INFO 12-08 13:24:26 [__init__.py:239] Automatically detected platform cuda.
INFO 12-08 13:24:26 [__init__.py:239] Automatically detected platform cuda.
INFO 12-08 13:24:27 [__init__.py:239] Automatically detected platform cuda.
INFO 12-08 13:24:27 [__init__.py:239] Automatically detected platform cuda.
INFO 12-08 13:24:27 [__init__.py:239] Automatically detected platform cuda.
INFO 12-08 13:24:27 [__init__.py:239] Automatically detected platform cuda.
INFO 12-08 13:24:27 [__init__.py:239] Automatically detected platform cuda.
INFO 12-08 13:24:27 [__init__.py:239] Automatically detected platform cuda.
Xattention Import Fail
Traceback (most recent call last):
  File "<frozen runpy>", line 189, in _run_module_as_main
  File "<frozen runpy>", line 112, in _get_module_details
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/__init__.py", line 31, in <module>
    from .modeling_flash_llama import PawLlamaForCausalLM
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/modeling_flash_llama.py", line 66, in <module>
    from sparseattn.src.Xattention import Xattention_prefill_dim3, Xattention_prefill_dim4
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/__init__.py", line 54, in <module>
    from .training import DistributedAttention
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/__init__.py", line 32, in <module>
    from .modeling_flash_qwen import PawQwen3ForCausalLM
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/modeling_flash_qwen.py", line 1914
    head_contrastive_loss = - （(jsd_total / float(count)) * 0.1）
                              ^
SyntaxError: invalid character '（' (U+FF08)
Xattention Import Fail
Traceback (most recent call last):
  File "<frozen runpy>", line 189, in _run_module_as_main
  File "<frozen runpy>", line 112, in _get_module_details
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/__init__.py", line 31, in <module>
    from .modeling_flash_llama import PawLlamaForCausalLM
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/modeling_flash_llama.py", line 66, in <module>
    from sparseattn.src.Xattention import Xattention_prefill_dim3, Xattention_prefill_dim4
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/__init__.py", line 54, in <module>
    from .training import DistributedAttention
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/__init__.py", line 32, in <module>
    from .modeling_flash_qwen import PawQwen3ForCausalLM
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/modeling_flash_qwen.py", line 1914
    head_contrastive_loss = - （(jsd_total / float(count)) * 0.1）
                              ^
SyntaxError: invalid character '（' (U+FF08)
Xattention Import Fail
Traceback (most recent call last):
  File "<frozen runpy>", line 189, in _run_module_as_main
  File "<frozen runpy>", line 112, in _get_module_details
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/__init__.py", line 31, in <module>
    from .modeling_flash_llama import PawLlamaForCausalLM
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/modeling_flash_llama.py", line 66, in <module>
    from sparseattn.src.Xattention import Xattention_prefill_dim3, Xattention_prefill_dim4
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/__init__.py", line 54, in <module>
    from .training import DistributedAttention
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/__init__.py", line 32, in <module>
    from .modeling_flash_qwen import PawQwen3ForCausalLM
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/modeling_flash_qwen.py", line 1914
    head_contrastive_loss = - （(jsd_total / float(count)) * 0.1）
                              ^
SyntaxError: invalid character '（' (U+FF08)
Xattention Import Fail
Traceback (most recent call last):
  File "<frozen runpy>", line 189, in _run_module_as_main
  File "<frozen runpy>", line 112, in _get_module_details
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/__init__.py", line 31, in <module>
    from .modeling_flash_llama import PawLlamaForCausalLM
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/modeling_flash_llama.py", line 66, in <module>
    from sparseattn.src.Xattention import Xattention_prefill_dim3, Xattention_prefill_dim4
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/__init__.py", line 54, in <module>
    from .training import DistributedAttention
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/__init__.py", line 32, in <module>
    from .modeling_flash_qwen import PawQwen3ForCausalLM
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/modeling_flash_qwen.py", line 1914
    head_contrastive_loss = - （(jsd_total / float(count)) * 0.1）
                              ^
SyntaxError: invalid character '（' (U+FF08)
Xattention Import Fail
Traceback (most recent call last):
  File "<frozen runpy>", line 189, in _run_module_as_main
  File "<frozen runpy>", line 112, in _get_module_details
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/__init__.py", line 31, in <module>
    from .modeling_flash_llama import PawLlamaForCausalLM
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/modeling_flash_llama.py", line 66, in <module>
    from sparseattn.src.Xattention import Xattention_prefill_dim3, Xattention_prefill_dim4
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/__init__.py", line 54, in <module>
    from .training import DistributedAttention
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/__init__.py", line 32, in <module>
    from .modeling_flash_qwen import PawQwen3ForCausalLM
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/modeling_flash_qwen.py", line 1914
    head_contrastive_loss = - （(jsd_total / float(count)) * 0.1）
                              ^
SyntaxError: invalid character '（' (U+FF08)
Xattention Import Fail
Xattention Import Fail
Traceback (most recent call last):
  File "<frozen runpy>", line 189, in _run_module_as_main
  File "<frozen runpy>", line 112, in _get_module_details
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/__init__.py", line 31, in <module>
    from .modeling_flash_llama import PawLlamaForCausalLM
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/modeling_flash_llama.py", line 66, in <module>
    from sparseattn.src.Xattention import Xattention_prefill_dim3, Xattention_prefill_dim4
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/__init__.py", line 54, in <module>
    from .training import DistributedAttention
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/__init__.py", line 32, in <module>
    from .modeling_flash_qwen import PawQwen3ForCausalLM
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/modeling_flash_qwen.py", line 1914
    head_contrastive_loss = - （(jsd_total / float(count)) * 0.1）
                              ^
SyntaxError: invalid character '（' (U+FF08)
Xattention Import Fail
Traceback (most recent call last):
  File "<frozen runpy>", line 189, in _run_module_as_main
  File "<frozen runpy>", line 112, in _get_module_details
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/__init__.py", line 31, in <module>
    from .modeling_flash_llama import PawLlamaForCausalLM
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/modeling_flash_llama.py", line 66, in <module>
    from sparseattn.src.Xattention import Xattention_prefill_dim3, Xattention_prefill_dim4
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/__init__.py", line 54, in <module>
    from .training import DistributedAttention
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/__init__.py", line 32, in <module>
    from .modeling_flash_qwen import PawQwen3ForCausalLM
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/modeling_flash_qwen.py", line 1914
    head_contrastive_loss = - （(jsd_total / float(count)) * 0.1）
                              ^
SyntaxError: invalid character '（' (U+FF08)
Traceback (most recent call last):
  File "<frozen runpy>", line 189, in _run_module_as_main
  File "<frozen runpy>", line 112, in _get_module_details
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/__init__.py", line 31, in <module>
    from .modeling_flash_llama import PawLlamaForCausalLM
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/modeling_flash_llama.py", line 66, in <module>
    from sparseattn.src.Xattention import Xattention_prefill_dim3, Xattention_prefill_dim4
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/__init__.py", line 54, in <module>
    from .training import DistributedAttention
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/__init__.py", line 32, in <module>
    from .modeling_flash_qwen import PawQwen3ForCausalLM
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/modeling_flash_qwen.py", line 1914
    head_contrastive_loss = - （(jsd_total / float(count)) * 0.1）
                              ^
SyntaxError: invalid character '（' (U+FF08)
W1208 13:24:30.075000 1293261 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 1293296 closing signal SIGTERM
W1208 13:24:30.078000 1293261 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 1293297 closing signal SIGTERM
W1208 13:24:30.080000 1293261 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 1293298 closing signal SIGTERM
W1208 13:24:30.082000 1293261 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 1293299 closing signal SIGTERM
W1208 13:24:30.084000 1293261 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 1293300 closing signal SIGTERM
W1208 13:24:30.089000 1293261 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 1293301 closing signal SIGTERM
W1208 13:24:30.091000 1293261 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 1293303 closing signal SIGTERM
E1208 13:24:30.680000 1293261 site-packages/torch/distributed/elastic/multiprocessing/api.py:869] failed (exitcode: 1) local_rank: 6 (pid: 1293302) of binary: /data1/anaconda3/envs/qqt/bin/python3
Traceback (most recent call last):
  File "/data1/anaconda3/envs/qqt/bin/torchrun", line 10, in <module>
    sys.exit(main())
             ^^^^^^
  File "/data1/anaconda3/envs/qqt/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 355, in wrapper
    return f(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^
  File "/data1/anaconda3/envs/qqt/lib/python3.11/site-packages/torch/distributed/run.py", line 918, in main
    run(args)
  File "/data1/anaconda3/envs/qqt/lib/python3.11/site-packages/torch/distributed/run.py", line 909, in run
    elastic_launch(
  File "/data1/anaconda3/envs/qqt/lib/python3.11/site-packages/torch/distributed/launcher/api.py", line 138, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data1/anaconda3/envs/qqt/lib/python3.11/site-packages/torch/distributed/launcher/api.py", line 269, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
training.lh_train_language_model FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2025-12-08_13:24:30
  host      : admin
  rank      : 6 (local_rank: 6)
  exitcode  : 1 (pid: 1293302)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
Traceback (most recent call last):
  File "<frozen runpy>", line 189, in _run_module_as_main
  File "<frozen runpy>", line 112, in _get_module_details
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/__init__.py", line 31, in <module>
    from .modeling_flash_llama import PawLlamaForCausalLM
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/modeling_flash_llama.py", line 66, in <module>
    from sparseattn.src.Xattention import Xattention_prefill_dim3, Xattention_prefill_dim4
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/__init__.py", line 102
    INFO 12-08 13:24:26 [__init__.py:239] Automatically detected platform cuda.
            ^
SyntaxError: leading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers
Traceback (most recent call last):
  File "<frozen runpy>", line 189, in _run_module_as_main
  File "<frozen runpy>", line 112, in _get_module_details
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/__init__.py", line 31, in <module>
    from .modeling_flash_llama import PawLlamaForCausalLM
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/modeling_flash_llama.py", line 66, in <module>
    from sparseattn.src.Xattention import Xattention_prefill_dim3, Xattention_prefill_dim4
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/__init__.py", line 102
    INFO 12-08 13:24:26 [__init__.py:239] Automatically detected platform cuda.
            ^
SyntaxError: leading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers
Traceback (most recent call last):
Traceback (most recent call last):
  File "<frozen runpy>", line 189, in _run_module_as_main
  File "<frozen runpy>", line 189, in _run_module_as_main
  File "<frozen runpy>", line 112, in _get_module_details
  File "<frozen runpy>", line 112, in _get_module_details
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/__init__.py", line 31, in <module>
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/__init__.py", line 31, in <module>
        from .modeling_flash_llama import PawLlamaForCausalLMfrom .modeling_flash_llama import PawLlamaForCausalLM

  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/modeling_flash_llama.py", line 66, in <module>
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/modeling_flash_llama.py", line 66, in <module>
        from sparseattn.src.Xattention import Xattention_prefill_dim3, Xattention_prefill_dim4from sparseattn.src.Xattention import Xattention_prefill_dim3, Xattention_prefill_dim4

  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/__init__.py", line 102
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/__init__.py", line 102
        INFO 12-08 13:24:26 [__init__.py:239] Automatically detected platform cuda.INFO 12-08 13:24:26 [__init__.py:239] Automatically detected platform cuda.

                        ^^

SyntaxErrorSyntaxError: : leading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integersleading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers

Traceback (most recent call last):
  File "<frozen runpy>", line 189, in _run_module_as_main
  File "<frozen runpy>", line 112, in _get_module_details
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/__init__.py", line 31, in <module>
    from .modeling_flash_llama import PawLlamaForCausalLM
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/modeling_flash_llama.py", line 66, in <module>
    from sparseattn.src.Xattention import Xattention_prefill_dim3, Xattention_prefill_dim4
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/__init__.py", line 102
    INFO 12-08 13:24:26 [__init__.py:239] Automatically detected platform cuda.
            ^
SyntaxError: leading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers
Traceback (most recent call last):
  File "<frozen runpy>", line 189, in _run_module_as_main
  File "<frozen runpy>", line 112, in _get_module_details
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/__init__.py", line 31, in <module>
    from .modeling_flash_llama import PawLlamaForCausalLM
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/modeling_flash_llama.py", line 66, in <module>
    from sparseattn.src.Xattention import Xattention_prefill_dim3, Xattention_prefill_dim4
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/__init__.py", line 102
    INFO 12-08 13:24:26 [__init__.py:239] Automatically detected platform cuda.
            ^
SyntaxError: leading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers
Traceback (most recent call last):
  File "<frozen runpy>", line 189, in _run_module_as_main
  File "<frozen runpy>", line 112, in _get_module_details
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/__init__.py", line 31, in <module>
    from .modeling_flash_llama import PawLlamaForCausalLM
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/modeling_flash_llama.py", line 66, in <module>
    from sparseattn.src.Xattention import Xattention_prefill_dim3, Xattention_prefill_dim4
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/__init__.py", line 102
    INFO 12-08 13:24:26 [__init__.py:239] Automatically detected platform cuda.
            ^
SyntaxError: leading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers
Traceback (most recent call last):
  File "<frozen runpy>", line 189, in _run_module_as_main
  File "<frozen runpy>", line 112, in _get_module_details
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/__init__.py", line 31, in <module>
    from .modeling_flash_llama import PawLlamaForCausalLM
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/modeling_flash_llama.py", line 66, in <module>
    from sparseattn.src.Xattention import Xattention_prefill_dim3, Xattention_prefill_dim4
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/__init__.py", line 102
    INFO 12-08 13:24:26 [__init__.py:239] Automatically detected platform cuda.
            ^
SyntaxError: leading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers
W1208 13:24:40.039000 1294469 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 1294509 closing signal SIGTERM
W1208 13:24:40.040000 1294469 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 1294510 closing signal SIGTERM
W1208 13:24:40.041000 1294469 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 1294511 closing signal SIGTERM
W1208 13:24:40.042000 1294469 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 1294512 closing signal SIGTERM
W1208 13:24:40.042000 1294469 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 1294513 closing signal SIGTERM
W1208 13:24:40.043000 1294469 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 1294514 closing signal SIGTERM
W1208 13:24:40.043000 1294469 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 1294517 closing signal SIGTERM
E1208 13:24:40.271000 1294469 site-packages/torch/distributed/elastic/multiprocessing/api.py:869] failed (exitcode: 1) local_rank: 6 (pid: 1294516) of binary: /data1/anaconda3/envs/qqt/bin/python3
Traceback (most recent call last):
  File "/data1/anaconda3/envs/qqt/bin/torchrun", line 10, in <module>
    sys.exit(main())
             ^^^^^^
  File "/data1/anaconda3/envs/qqt/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 355, in wrapper
    return f(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^
  File "/data1/anaconda3/envs/qqt/lib/python3.11/site-packages/torch/distributed/run.py", line 918, in main
    run(args)
  File "/data1/anaconda3/envs/qqt/lib/python3.11/site-packages/torch/distributed/run.py", line 909, in run
    elastic_launch(
  File "/data1/anaconda3/envs/qqt/lib/python3.11/site-packages/torch/distributed/launcher/api.py", line 138, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data1/anaconda3/envs/qqt/lib/python3.11/site-packages/torch/distributed/launcher/api.py", line 269, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
training.lh_train_language_model FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2025-12-08_13:24:40
  host      : admin
  rank      : 6 (local_rank: 6)
  exitcode  : 1 (pid: 1294516)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
Traceback (most recent call last):
  File "<frozen runpy>", line 189, in _run_module_as_main
  File "<frozen runpy>", line 112, in _get_module_details
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/__init__.py", line 31, in <module>
    from .modeling_flash_llama import PawLlamaForCausalLM
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/modeling_flash_llama.py", line 66, in <module>
    from sparseattn.src.Xattention import Xattention_prefill_dim3, Xattention_prefill_dim4
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/__init__.py", line 102
    INFO 12-08 13:24:26 [__init__.py:239] Automatically detected platform cuda.
            ^
SyntaxError: leading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers
Traceback (most recent call last):
  File "<frozen runpy>", line 189, in _run_module_as_main
  File "<frozen runpy>", line 112, in _get_module_details
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/__init__.py", line 31, in <module>
    from .modeling_flash_llama import PawLlamaForCausalLM
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/modeling_flash_llama.py", line 66, in <module>
    from sparseattn.src.Xattention import Xattention_prefill_dim3, Xattention_prefill_dim4
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/__init__.py", line 102
    INFO 12-08 13:24:26 [__init__.py:239] Automatically detected platform cuda.
            ^
SyntaxError: leading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers
Traceback (most recent call last):
  File "<frozen runpy>", line 189, in _run_module_as_main
  File "<frozen runpy>", line 112, in _get_module_details
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/__init__.py", line 31, in <module>
    from .modeling_flash_llama import PawLlamaForCausalLM
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/modeling_flash_llama.py", line 66, in <module>
    from sparseattn.src.Xattention import Xattention_prefill_dim3, Xattention_prefill_dim4
Traceback (most recent call last):
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/__init__.py", line 102
    INFO 12-08 13:24:26 [__init__.py:239] Automatically detected platform cuda.
       File "<frozen runpy>", line 189, in _run_module_as_main
        File "<frozen runpy>", line 112, in _get_module_details
 ^
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/__init__.py", line 31, in <module>
SyntaxError: leading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers
    from .modeling_flash_llama import PawLlamaForCausalLM
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/modeling_flash_llama.py", line 66, in <module>
    from sparseattn.src.Xattention import Xattention_prefill_dim3, Xattention_prefill_dim4
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/__init__.py", line 102
    INFO 12-08 13:24:26 [__init__.py:239] Automatically detected platform cuda.
            ^
SyntaxError: leading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers
Traceback (most recent call last):
  File "<frozen runpy>", line 189, in _run_module_as_main
  File "<frozen runpy>", line 112, in _get_module_details
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/__init__.py", line 31, in <module>
    from .modeling_flash_llama import PawLlamaForCausalLM
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/modeling_flash_llama.py", line 66, in <module>
    from sparseattn.src.Xattention import Xattention_prefill_dim3, Xattention_prefill_dim4
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/__init__.py", line 102
    INFO 12-08 13:24:26 [__init__.py:239] Automatically detected platform cuda.
            ^
SyntaxError: leading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers
Traceback (most recent call last):
  File "<frozen runpy>", line 189, in _run_module_as_main
  File "<frozen runpy>", line 112, in _get_module_details
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/__init__.py", line 31, in <module>
    from .modeling_flash_llama import PawLlamaForCausalLM
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/modeling_flash_llama.py", line 66, in <module>
    from sparseattn.src.Xattention import Xattention_prefill_dim3, Xattention_prefill_dim4
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/__init__.py", line 102
    INFO 12-08 13:24:26 [__init__.py:239] Automatically detected platform cuda.
            ^
SyntaxError: leading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers
Traceback (most recent call last):
  File "<frozen runpy>", line 189, in _run_module_as_main
  File "<frozen runpy>", line 112, in _get_module_details
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/__init__.py", line 31, in <module>
    from .modeling_flash_llama import PawLlamaForCausalLM
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/modeling_flash_llama.py", line 66, in <module>
    from sparseattn.src.Xattention import Xattention_prefill_dim3, Xattention_prefill_dim4
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/__init__.py", line 102
    INFO 12-08 13:24:26 [__init__.py:239] Automatically detected platform cuda.
            ^
SyntaxError: leading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers
Traceback (most recent call last):
  File "<frozen runpy>", line 189, in _run_module_as_main
  File "<frozen runpy>", line 112, in _get_module_details
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/__init__.py", line 31, in <module>
    from .modeling_flash_llama import PawLlamaForCausalLM
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/training/modeling_flash_llama.py", line 66, in <module>
    from sparseattn.src.Xattention import Xattention_prefill_dim3, Xattention_prefill_dim4
  File "/data1/lcm_lab/qqt/SparseAttn/sparseattn/__init__.py", line 102
    INFO 12-08 13:24:26 [__init__.py:239] Automatically detected platform cuda.
            ^
SyntaxError: leading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers
W1208 13:24:49.689000 1295069 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 1295115 closing signal SIGTERM
W1208 13:24:49.689000 1295069 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 1295117 closing signal SIGTERM
W1208 13:24:49.690000 1295069 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 1295118 closing signal SIGTERM
W1208 13:24:49.690000 1295069 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 1295119 closing signal SIGTERM
W1208 13:24:49.691000 1295069 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 1295120 closing signal SIGTERM
W1208 13:24:49.691000 1295069 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 1295121 closing signal SIGTERM
W1208 13:24:49.692000 1295069 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 1295122 closing signal SIGTERM
E1208 13:24:49.919000 1295069 site-packages/torch/distributed/elastic/multiprocessing/api.py:869] failed (exitcode: 1) local_rank: 1 (pid: 1295116) of binary: /data1/anaconda3/envs/qqt/bin/python3
Traceback (most recent call last):
  File "/data1/anaconda3/envs/qqt/bin/torchrun", line 10, in <module>
    sys.exit(main())
             ^^^^^^
  File "/data1/anaconda3/envs/qqt/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 355, in wrapper
    return f(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^
  File "/data1/anaconda3/envs/qqt/lib/python3.11/site-packages/torch/distributed/run.py", line 918, in main
    run(args)
  File "/data1/anaconda3/envs/qqt/lib/python3.11/site-packages/torch/distributed/run.py", line 909, in run
    elastic_launch(
  File "/data1/anaconda3/envs/qqt/lib/python3.11/site-packages/torch/distributed/launcher/api.py", line 138, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data1/anaconda3/envs/qqt/lib/python3.11/site-packages/torch/distributed/launcher/api.py", line 269, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
training.lh_train_language_model FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2025-12-08_13:24:49
  host      : admin
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 1295116)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
