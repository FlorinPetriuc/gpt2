#!/usr/bin/python3

import argparse
import gpt2
import logging
import math
import tiktoken
import torch

logging.basicConfig(level=logging.INFO)
device = "cuda" if torch.cuda.is_available() else "cpu"
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
logging.info(f"Using device {device}")
logging.info(f"Using bf16: %s", use_bf16)

def train(model, data_path):
    logging.info("Training using %s", data_path)
    trainer = gpt2.Trainer(model, data_path, device, use_bf16)
    trainer.start()

@torch.no_grad()
def generate(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    while True:
        data = input("$ ")
        tokens = torch.tensor([enc.encode_ordinary(data)], device=device)
        out = model.generate(tokens, max_new_tokens=200, temperature=0.8, top_k=50)
        print(enc.decode(out[0].tolist()))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=gpt2.GPT2.SIZES.keys(), default="small")
    parser.add_argument("--train-data")
    parser.add_argument("--generate")
    args = parser.parse_args()
    logging.info(f"Using size {args.size}")
    model = gpt2.GPT2(args.size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info("GTP-2 %s: %sM parameters on %s", args.size, n_params / 1e6, device)
    if args.train_data:
        train(model, args.train_data)
    elif args.generate:
        generate(model, args.generate)

if __name__ == "__main__":
    main()
