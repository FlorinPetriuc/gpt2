#!/usr/bin/python3

from pathlib import Path
import block
import contextlib
import logging
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPT2(nn.Module):
    VOCAB_SIZE = 50257
    BLOCK_SIZE = 1024
    DROPOUT = 0.0
    BIAS = True
    SIZES = {
        "small":  { "n_layer": 12, "n_head": 12, "n_embd":  768 },
        "medium": { "n_layer": 24, "n_head": 16, "n_embd": 1024 },
        "large":  { "n_layer": 36, "n_head": 20, "n_embd": 1280 },
        "xl":     { "n_layer": 48, "n_head": 25, "n_embd": 1600 },
    }

    def __init__(self, size):
        super().__init__()
        self._size = self.SIZES[size]
        self.wte = nn.Embedding(self.VOCAB_SIZE, self._size["n_embd"])
        self.wpe = nn.Embedding(self.BLOCK_SIZE, self._size["n_embd"])
        self.drop = nn.Dropout(self.DROPOUT)
        self.h = nn.ModuleList(
            block.Block(
                self._size["n_embd"],
                self._size["n_head"],
                self.DROPOUT,
                self.BIAS
            ) for _ in range(self._size["n_layer"])
        )
        self.ln_f = nn.LayerNorm(self._size["n_embd"])
        self.lm_head = nn.Linear(self._size["n_embd"], self.VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)
        resid_std = 0.02 / math.sqrt(2 * self._size["n_layer"])
        for blk in self.h:
            nn.init.normal_(blk.attn.c_proj.weight, std=resid_std)
            nn.init.normal_(blk.mlp.c_proj.weight, std=resid_std)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.BLOCK_SIZE
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.h:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
        return idx

class Trainer:
    def __init__(self, model, data_path, device, use_bf16):
        torch.set_float32_matmul_precision("high")
        self._model = model
        self._data_path = data_path
        self._device = device
        self._loss_hist, self._lr_hist, self._eval_hist = [], [], []
        if use_bf16:
            self._ctx = torch.autocast(self._device, dtype=torch.bfloat16)
        else:
            self._ctx = contextlib.nullcontext()
        self._WEIGHT_DECAY = 0.1
        self._LR = 6e-4
        self._MIN_LR = 6e-5
        self._WARMUP = 100
        self._EPOCHS = 1000
        self._EVAL_EVERY = 100
        self._EVAL_ITERS = 20
        self._BATCH_SIZE = 4
        self._SEQ_LEN = 256
        self._GRAD_CLIP = 1.0
        self._GRAD_ACCUM = 4

    def _tokenize_data_to_file(self):
        enc = tiktoken.get_encoding("gpt2")
        out = Path(self._data_path).with_suffix(".bin")
        n_tokens = 0
        with open(self._data_path, "r") as fd_in, open(out, "wb") as fd_out:
            while chunk := fd_in.read(64 * 1024 * 1024):
                chunk += fd_in.readline()
                ids = np.array(enc.encode_ordinary(chunk), dtype=np.uint16)
                ids.tofile(fd_out)
                n_tokens += len(ids)
        logging.info("Loaded %s tokens is %s", n_tokens, out)
        return out

    def _load_data(self):
        bin_file = self._tokenize_data_to_file()
        data = np.memmap(bin_file, dtype=np.uint16, mode="r")
        div = int(0.9 * len(data))
        logging.info("Training on %s tokens, validating on %s tokens", div, len(data) - div)
        self._train_data = data[:div]
        self._val_data = data[div:]
        self._model_file = Path(bin_file).with_suffix(".nn")
        self._result_fig = Path(bin_file).with_suffix(".png")

    def _adjusted_lr(self, epoch):
        if epoch < self._WARMUP:
            return self._LR * (epoch + 1) / self._WARMUP
        if epoch >= self._EPOCHS:
            return self._MIN_LR
        ratio = (epoch - self._WARMUP) / (self._EPOCHS - self._WARMUP)
        return self._MIN_LR + 0.5 * (1 + math.cos(math.pi * ratio)) * (self._LR - self._MIN_LR)

    def _get_batch(self, data):
        ix = torch.randint(len(data) - self._SEQ_LEN - 1, (self._BATCH_SIZE,)).tolist()
        x = torch.stack([torch.from_numpy(data[i : i + self._SEQ_LEN].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + self._SEQ_LEN].astype(np.int64)) for i in ix])
        return x.to(self._device), y.to(self._device)

    @torch.no_grad()
    def _estimate_losses(self):
        self._model.eval()
        out = {}
        for type, data in {"train": self._train_data, "val": self._val_data}.items():
            losses = torch.zeros(self._EVAL_ITERS)
            for i in range(self._EVAL_ITERS):
                x, y = self._get_batch(data)
                with self._ctx:
                    _, loss = self._model(x, y)
                losses[i] = loss.item()
            out[type] = losses.mean().item()
        self._model.train()
        return out

    def _save_model(self, epoch, loss):
        torch.save(
            { "model": self._model.state_dict(), "size": self._model._size, "epoch": epoch, "val_loss": loss },
            self._model_file
        )
        logging.info("Saved %s", self._model_file)

    def _do_training(self, optim):
        best_val = float("inf")
        for epoch in range(self._EPOCHS + 1):
            lr = self._adjusted_lr(epoch)
            for group in optim.param_groups:
                group["lr"] = lr
            if epoch % self._EVAL_EVERY == 0 or epoch == self._EPOCHS:
                losses = self._estimate_losses()
                logging.info("epoch %s: train %s, val: %s", epoch, losses["train"], losses["val"])
                self._eval_hist.append((epoch, losses["train"], losses["val"]))
                if losses["val"] < best_val:
                    best_val = losses["val"]
                    self._save_model(epoch, best_val)
            if epoch < self._EPOCHS:
                optim.zero_grad(set_to_none=True)
                loss_accum = 0.0
                for _ in range(self._GRAD_ACCUM):
                    x, y = self._get_batch(self._train_data)
                    with self._ctx:
                        _, loss = self._model(x, y)
                    loss /= self._GRAD_ACCUM
                    loss_accum += loss.item()
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._GRAD_CLIP)
                optim.step()
                self._loss_hist.append(loss_accum)
                self._lr_hist.append(lr)
                if epoch % 10 == 0:
                    logging.info("epoch %s: loss %s, lr %s", epoch, loss_accum, lr)

    @torch.no_grad()
    def _do_test(self):
        enc = tiktoken.get_encoding("gpt2")
        self._model.eval()
        prompt = torch.tensor([enc.encode_ordinary("Who are you?")], device=self._device)
        with self._ctx:
            out = self._model.generate(prompt, max_new_tokens=200, temperature=0.8, top_k = 50)
        logging.info("Sample: %s", enc.decode(out[0].tolist()))

    def _plot_results(self):
        matplotlib.use("Agg")
        epochs = range(len(self._loss_hist))
        fig, (ax_loss, ax_lr) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        ax_loss.plot(epochs, self._loss_hist, alpha=0.35, label="train (per epoch)")
        if self._eval_hist:
            ev = np.array(self._eval_hist)
            ax_loss.plot(ev[:, 0], ev[:, 1], "o-", label="train (eval)")
            ax_loss.plot(ev[:, 0], ev[:, 2], "o-", label="val")
        ax_loss.set_title("Loss")
        ax_loss.set_ylabel("loss")
        ax_loss.grid(alpha=0.3)
        ax_lr.plot(epochs, self._lr_hist)
        ax_lr.set_xlabel("step")
        ax_lr.set_ylabel("learning rate")
        ax_lr.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self._result_fig, dpi=150)
        plt.close(fig)
        logging.info("Saved %s", self._result_fig)

    def start(self):
        self._load_data()
        decay = [p for p in self._model.parameters() if p.dim() >= 2]
        no_decay = [p for p in self._model.parameters() if p.dim() < 2]
        optim = torch.optim.AdamW(
            [{"params": decay, "weight_decay": self._WEIGHT_DECAY}, {"params": no_decay, "weight_decay": 0.0}],
            lr = self._LR,
            betas=(0.9, 0.95),
            fused = torch.cuda.is_available()
        )
        self._do_training(optim)
        self._plot_results()
        self._do_test()
