import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from moba.moba import register_moba, MoBAConfig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/data1/hf_model/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--moba-chunk-size", type=int, default=4096)
    parser.add_argument("--moba-topk", type=int, default=12)
    parser.add_argument(
        "--attn",
        default="moba",
        help="choose attention backend",
        choices=["flash_attention_2", "moba", "moba_naive"],
    )
    args = parser.parse_args()

    args.moba_chunk_size = 1024
    args.moba_topk = 8
    args.model = "/data1/hf_model/Meta-Llama-3.1-8B-Instruct"
    args.attn = "moba"
    register_moba(MoBAConfig(args.moba_chunk_size, args.moba_topk))
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation=args.attn,
    )

    tknz = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)


    prompt = "Artificial intelligence is transforming the world in profound ways, from healthcare and education to transportation and entertainment. Researchers continue to push the boundaries of what machines can learn and understand, raising both exciting opportunities and important ethical questions about the future of human-AI collaboration."
    input_tokens = tknz.encode(prompt)
    input_ids = torch.tensor([input_tokens], device=model.device)
    tokens = model.generate(input_ids, max_length=60, do_sample=False)
    print(tokens)
    print(tknz.decode(tokens.squeeze().tolist()))
