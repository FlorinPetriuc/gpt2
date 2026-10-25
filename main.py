#!/usr/bin/python3

import argparse
import gpt2
import logging
import math
import matplotlib
import matplotlib.pyplot as plt
import tiktoken
import torch

matplotlib.use("Agg")

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

def print_predictions(logits, enc):
    probs = logits[0].float().cpu().softmax(dim=-1)
    top_p, top_t = probs[-1].topk(5)
    for prob, t in zip(top_p.tolist(), top_t.tolist()):
        logging.info("%s with probability %s%%", enc.decode([t]), prob * 100)

def plot_overview(activations, tokens, out):
    mean_abs = activations.abs().mean(dim=2).numpy()
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(tokens)), 6))
    im = ax.imshow(mean_abs, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_ylabel("layer")
    ax.set_title("Mean |MLP activation| per layer and token")
    fig.colorbar(im, label="mean |activation|")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logging.info("Saved %s", out)

def plot_layer(activations, tokens, out):
    a = activations[-1]
    top_k = 5
    top = a.abs().max(dim=0).values.topk(top_k).indices
    data = a[:, top].numpy().T
    m = abs(data).max()
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(tokens)), 0.2 * top_k + 2))
    im = ax.imshow(data, aspect="auto", cmap="coolwarm", vmin=-m, vmax=m)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([str(i) for i in top.tolist()], fontsize=6)
    ax.set_ylabel("neuron index")
    ax.set_title(f"Layer {len(activations.shape) - 1}: top {top_k} neurons by peak activation")
    fig.colorbar(im, label="activation")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close()
    logging.info("Saved %s", out)

@torch.no_grad()
def predict(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    model.activate_prehooks()
    query = 0
    while True:
        data = input("$ ")
        ids = enc.encode_ordinary(data)[:model.BLOCK_SIZE]
        tokens = [repr(enc.decode([t])) for t in ids]
        idx = torch.tensor([ids], device=device)
        model.reset_activations()
        logits, _ = model(idx)
        print_predictions(logits, enc)
        plot_overview(model.get_activations(), tokens, "query-" + str(query) + "-overview.png")
        plot_layer(model.get_activations(), tokens, "guery-" + str(query) + "-layer.png")
        query += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=gpt2.GPT2.SIZES.keys(), default="small")
    parser.add_argument("--train-data")
    parser.add_argument("--generate")
    parser.add_argument("--predict")
    args = parser.parse_args()
    logging.info(f"Using size {args.size}")
    model = gpt2.GPT2(args.size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info("GPT-2 %s: %sM parameters on %s", args.size, n_params / 1e6, device)
    if args.train_data:
        train(model, args.train_data)
    elif args.generate:
        generate(model, args.generate)
    elif args.predict:
        predict(model, args.predict)

if __name__ == "__main__":
    main()
