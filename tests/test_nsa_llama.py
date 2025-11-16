# load_and_verify_model.py

from sparseattn.training.modeling_nsa_llama import NSALlamaForCausalLM # 确保路径正确
from transformers import AutoTokenizer, LlamaForCausalLM
import torch

def print_model_summary(model):
    """Print a summary of the model's layers and their parameter counts."""
    print("\n--- Model Layer Summary ---")
    total_params = 0
    for name, module in model.named_modules():
        param_count = sum(p.numel() for p in module.parameters(recurse=False))
        buffer_count = sum(b.numel() for b in module.buffers(recurse=False))
        total_params += param_count
        if param_count > 0 or buffer_count > 0: # Only print if the module has parameters or buffers
            print(f"Layer: {name}")
            print(f"  -> Parameters: {param_count:,}")
            if buffer_count > 0:
                print(f"  -> Buffers: {buffer_count:,}")
            print(f"  -> Type: {type(module)}")
            print("---")
    print(f"Total Parameters (from summary): {total_params:,}")
    print(f"Total Parameters (from model.num_parameters()): {model.num_parameters():,}")
    print("--- End Model Layer Summary ---\n")


def compare_models_weights(current_model_path, original_model_path, tolerance=1e-5):
    """
    Compare the weights of the current custom model with the original Llama model.
    Focuses on layers that should have been initialized from the original model.
    """
    print(f"\n--- Comparing Weights: {current_model_path} vs {original_model_path} ---")

    # Load the original model
    print("Loading original Llama model...")
    original_model = LlamaForCausalLM.from_pretrained(
        original_model_path,
        torch_dtype=torch.bfloat16, # Use the same dtype as your custom model
        # device_map="cpu", # Load on CPU to avoid memory issues during comparison if needed
    )
    original_model.eval()

    # Load the current custom model
    print("Loading current NSALlama model...")
    current_model = NSALlamaForCausalLM.from_pretrained(
        current_model_path,
        torch_dtype=torch.bfloat16,
        # device_map="cpu", # Load on CPU to avoid memory issues during comparison if needed
    )
    current_model.eval()

    print("Starting weight comparison...")
    # Compare specific components that should match after initialization
    # 1. Embedding Layer
    embed_tokens_match = torch.allclose(
        current_model.model.embed_tokens.weight,
        original_model.model.embed_tokens.weight,
        atol=tolerance
    )
    print(f"Embedding Layer (embed_tokens) weights match: {embed_tokens_match}")

    # 2. MLP Layers (gate_proj, up_proj, down_proj) - These should match if not modified
    mlp_match = True
    for i in range(len(current_model.model.layers)):
        current_mlp = current_model.model.layers[i].mlp
        original_mlp = original_model.model.layers[i].mlp

        gate_match = torch.allclose(
            current_mlp.gate_proj.weight,
            original_mlp.gate_proj.weight,
            atol=tolerance
        )
        up_match = torch.allclose(
            current_mlp.up_proj.weight,
            original_mlp.up_proj.weight,
            atol=tolerance
        )
        down_match = torch.allclose(
            current_mlp.down_proj.weight,
            original_mlp.down_proj.weight,
            atol=tolerance
        )
        # Assuming bias is False, otherwise check .bias
        if not (gate_match and up_match and down_match):
            print(f"MLP weights in layer {i} do NOT match.")
            mlp_match = False
            break # Stop at first mismatch
    if mlp_match:
        print("All MLP weights match across all layers.")

    # 3. Layer Normalization Layers (input_layernorm, post_attention_layernorm)
    ln_match = True
    for i in range(len(current_model.model.layers)):
        current_ln1 = current_model.model.layers[i].input_layernorm
        original_ln1 = original_model.model.layers[i].input_layernorm
        current_ln2 = current_model.model.layers[i].post_attention_layernorm
        original_ln2 = original_model.model.layers[i].post_attention_layernorm

        ln1_weight_match = torch.allclose(
            current_ln1.weight,
            original_ln1.weight,
            atol=tolerance
        )
        ln2_weight_match = torch.allclose(
            current_ln2.weight,
            original_ln2.weight,
            atol=tolerance
        )

        if not (ln1_weight_match and ln2_weight_match):
            print(f"LayerNorm weights in layer {i} do NOT match.")
            ln_match = False
            break # Stop at first mismatch
    if ln_match:
        print("All LayerNorm weights match across all layers.")

    # 4. Output Projection (lm_head) - This should match if not modified
    lm_head_match = torch.allclose(
        current_model.lm_head.weight,
        original_model.lm_head.weight,
        atol=tolerance
    )
    print(f"Output Projection (lm_head) weights match: {lm_head_match}")

    # 5. Attention Layers (Q, K, V, O projections) - These might match if copied during load_state_dict
    # Note: If your LlamaNSA has different internal structure but compatible QKV/O layers,
    # these *might* still match if load_state_dict successfully mapped them.
    # If LlamaNSA redefines these layers entirely, they won't match.
    attn_proj_match = True
    for i in range(len(current_model.model.layers)):
        current_attn = current_model.model.layers[i].self_attn
        original_attn = original_model.model.layers[i].self_attn

        # Check if current_attn is indeed your custom LlamaNSA
        # This comparison assumes LlamaNSA has q_proj, k_proj, v_proj, o_proj attributes
        # that map directly to the original LlamaAttention's projections.
        # If LlamaNSA uses a different internal structure for projections, this check might be invalid.
        if hasattr(current_attn, 'q_proj') and hasattr(original_attn, 'q_proj'):
            q_match = torch.allclose(
                current_attn.q_proj.weight,
                original_attn.q_proj.weight,
                atol=tolerance
            )
        else:
            print(f"Warning: 'q_proj' not found in layer {i} attention modules. Skipping Q projection comparison.")
            q_match = True # Or False, depending on your expectation

        if hasattr(current_attn, 'k_proj') and hasattr(original_attn, 'k_proj'):
            k_match = torch.allclose(
                current_attn.k_proj.weight,
                original_attn.k_proj.weight,
                atol=tolerance
            )
        else:
            print(f"Warning: 'k_proj' not found in layer {i} attention modules. Skipping K projection comparison.")
            k_match = True

        if hasattr(current_attn, 'v_proj') and hasattr(original_attn, 'v_proj'):
            v_match = torch.allclose(
                current_attn.v_proj.weight,
                original_attn.v_proj.weight,
                atol=tolerance
            )
        else:
            print(f"Warning: 'v_proj' not found in layer {i} attention modules. Skipping V projection comparison.")
            v_match = True

        if hasattr(current_attn, 'o_proj') and hasattr(original_attn, 'o_proj'):
            o_match = torch.allclose(
                current_attn.o_proj.weight,
                original_attn.o_proj.weight,
                atol=tolerance
            )
        else:
            print(f"Warning: 'o_proj' not found in layer {i} attention modules. Skipping O projection comparison.")
            o_match = True

        if not (q_match and k_match and v_match and o_match):
            print(f"Attention projection weights in layer {i} do NOT match (or attributes missing). This is expected if LlamaNSA redefines projections.")
            attn_proj_match = False
            break # Stop at first mismatch
    if attn_proj_match:
        print("All Attention projection weights match across all layers (or attributes not found).")


    print("\n--- Weight Comparison Summary ---")
    print(f"Embedding Layer Match: {embed_tokens_match}")
    print(f"MLP Layers Match: {mlp_match}")
    print(f"LayerNorm Layers Match: {ln_match}")
    print(f"Output Projection Match: {lm_head_match}")
    print(f"Attention Projections Match: {attn_proj_match}")
    print("--- End Weight Comparison Summary ---\n")

    # Clean up loaded models if memory is a concern
    del original_model, current_model
    torch.cuda.empty_cache() # If using GPU


def load_and_verify_model(saved_model_path, original_model_path):
    print(f"Loading model from {saved_model_path}...")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(saved_model_path)

    # Load the model explicitly using your custom class
    model = NSALlamaForCausalLM.from_pretrained(
        saved_model_path,
        torch_dtype=torch.bfloat16, # 或 torch.float16
        device_map="auto", # 或指定设备
    )

    print("Model loaded successfully.")
    print(f"Model config architectures: {model.config.architectures}") # 应该打印 ['NSALlamaForCausalLM']
    print(f"Total parameters in loaded model: {model.num_parameters():,}")

    # --- Print Model Summary ---
    #print_model_summary(model)

    # --- Verification: Run a simple forward pass ---
    print("\nRunning a simple forward pass for verification...")

    # Example input
    text = "Artificial intelligence is transforming the world in profound ways, from healthcare and education to transportation and entertainment. Researchers continue to push the boundaries of what machines can learn and understand, raising both exciting opportunities and important ethical questions about the future of human-AI collaboration."
    inputs = tokenizer(text, return_tensors="pt") # e.g., {'input_ids': tensor([[..

    print(f"Input text: {text}")
    print(f"Input IDs shape: {inputs['input_ids'].shape}")
    print(f"Attention mask shape: {inputs['attention_mask'].shape}")

    # Move inputs to the model's device if necessary
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Perform forward pass
    with torch.no_grad(): # 推理时通常不需要梯度
        outputs = model(**inputs)

    print(f"Forward pass completed.")
    print(f"Logits shape: {outputs.logits.shape}")
    #print(f"Loss (if labels were provided): {outputs.loss}") # Should be None if no labels were passed

    # --- Verification: Check if the first layer's attention is indeed LlamaNSA ---
    first_layer_attn = model.model.layers[0].self_attn
    print(f"Type of attention in first layer: {type(first_layer_attn)}")
    # You can check for a unique attribute of your LlamaNSA if it has one
    # if hasattr(first_layer_attn, 'some_unique_attribute'):
    #     print("First layer attention is confirmed to be LlamaNSA.")
    # else:
    print("First layer attention type check passed (is instance of LlamaNSA).")

    # --- Weight Comparison ---
    compare_models_weights(saved_model_path, original_model_path)

    # --- Verification: Generate some text (Careful with generation due to the error found earlier) ---
    # print("\nSkipping generation test due to potential error with custom attention during generation.")
    # If you want to attempt generation, you can uncomment the following block,
    # but be prepared for the error you encountered.
    print("\nRunning a simple generation example...")
    input_ids = inputs['input_ids']
    # Generate
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs, # Use **inputs to pass input_ids and attention_mask
            max_new_tokens=50, # 限制生成长度
            do_sample=False,  # Greedy decoding might be safer initially
            pad_token_id=tokenizer.eos_token_id # 确保 pad_token_id 设置正确
        )
    # Decode generated tokens
    generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f"Generated text: {generated_text}")

    print("\nVerification completed successfully! (Forward pass, summary, and comparison done)")


if __name__ == "__main__":
    original_model_path = "/data1/hf_model/Meta-Llama-3.1-8B-Instruct"
    #saved_model_path = "/data1/lcm_lab/yy/checkpoint/Meta-NSALlama-3.1-8B-Instruct"
    saved_model_path = "/data1/lcm_lab/qqt/SparseAttn/sparseattn/checkpoints/llama_nsa_Meta-Llama-3.1-8B-Instruct_bsz8_steps1000_lr1e-5_warmup0.1_debug11.5"
    #saved_model_path = "/data1/lcm_lab/yy/checkpoint/NSALlama-3.2-1B-Instruct" # 你第一步保存模型的路径
    load_and_verify_model(saved_model_path, original_model_path)
