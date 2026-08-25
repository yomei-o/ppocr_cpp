"""Fine-tune the recogniser in Python, writing the same weight file the C++ side reads.

  python tools/train_ref.py --data scratch/recdata/train.txt --val scratch/recdata/val.txt \
                            --steps 60 --batch 4 --lr 2e-4 --eval-every 20 --out ft.bin

  ./ppocr.exe run --img photo.jpg --weights ft.bin      # the C++ side uses what this wrote

WHY THIS EXISTS. `ppocr train` (pure/train_rec.hpp) runs a hand-written reverse-mode autograd over
the ONNX graph. This does the same job through PyTorch's autograd instead, which makes the two a
cross-check rather than a duplicate: if the hand-written backward has a sign or an axis wrong, the
two runs disagree. `tools/grad_parity.py` compares them gradient by gradient.

The file format is shared on purpose. Fine-tuning in Python and recognising in C++ (or the reverse)
has to work, or "both languages can do it" is two separate features that happen to share a name.

WHAT IS DELIBERATELY THE SAME AS THE C++:

  * the trainable set is chosen by graph position -- Conv/Gemm weights and biases, MatMul weights,
    BatchNorm gamma and beta, and Add/Mul constants with exactly one non-1 dimension. Not "every
    float constant": that sweeps up literals like the 2.0 in a Pow, and an optimiser that nudges an
    exponent to 2.0003 turns pow(negative, 2.0003) into NaN. This picks 68 tensors in both languages.

  * BatchNorm is frozen. .eval() keeps the running statistics as constants while gamma and beta
    still take gradients, which is what the C++ does and what fine-tuning on batches of 4 needs.

  * a batch is `batch` separate forward passes. The recogniser's input width follows the aspect
    ratio, so padding a batch to its widest member changes the picture the model sees.

  * the loss is CTC over log(probs) with the probabilities floored at 1e-7. The graph ends in a
    Softmax, so the log is taken after it rather than using log_softmax. A lower floor is not safer:
    d/dv log v is 1/v, so a floor of 1e-12 puts 1e12 into the backward pass.

ONNX PLUMBING. Two things about this export need handling before onnx2torch will touch it. Paddle
writes the weights as Constant *nodes* rather than graph initializers, and onnx2torch looks for
initializers; and once converted, a tensor's original ONNX name is gone, while the weight file is
keyed by that name. Both are dealt with in prepare() and map_params() below.
"""
import argparse
import hashlib
import os
import random
import struct
import sys

import cv2
import numpy as np
import onnx
import torch
from onnx import numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppocr_ref import ctc_greedy, load_dict, rec_input  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC = os.path.join(ROOT, "models", "ppocrv5-mobile-rec.onnx")
DICT = os.path.join(ROOT, "models", "ppocrv5_dict.txt")

MAGIC = b"PPOCRW1\0"


# ---- the graph -------------------------------------------------------------------------------
def prepare(path):
    """Load the .onnx and move Constant nodes into graph.initializer.

    PaddleOCR's exporter emits `Constant` nodes for the weights. The C++ interpreter treats the two
    the same, so it never noticed; onnx2torch's converters index graph.initializers directly and
    raise KeyError on the first Conv. Rewriting the model in memory is enough -- nothing on disk
    changes, and the values and names are carried over untouched.
    """
    model = onnx.load(path)
    g = model.graph
    keep, moved = [], 0
    for n in g.node:
        value = next((a.t for a in n.attribute if a.name == "value"), None)
        if n.op_type == "Constant" and len(n.output) == 1 and value is not None:
            t = onnx.TensorProto()
            t.CopyFrom(value)
            t.name = n.output[0]
            g.initializer.append(t)
            moved += 1
        else:
            keep.append(n)
    del g.node[:]
    g.node.extend(keep)
    return model, moved


def select(graph):
    """The names of the tensors to train, by graph position. Mirrors onx::Model::trainable()."""
    init = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    names = []

    def take(node, idx):
        if len(node.input) > idx and node.input[idx] in init:
            names.append(node.input[idx])

    for n in graph.node:
        if n.op_type in ("Conv", "ConvTranspose", "Gemm"):
            take(n, 1)
            take(n, 2)
        elif n.op_type == "MatMul":
            take(n, 1)
        elif n.op_type == "BatchNormalization":
            take(n, 1)  # gamma
            take(n, 2)  # beta
        elif n.op_type in ("Add", "Mul"):
            # A bias or a per-channel scale: a constant broadcast along exactly one axis. A scalar
            # (every dim 1) is a literal like the 2.0 in a Pow and must not be optimised.
            for v in n.input:
                if v in init and sum(1 for d in init[v].shape if d != 1) == 1:
                    names.append(v)
    return list(dict.fromkeys(names)), init


def _digest(a):
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()


def map_params(torch_model, wanted, init):
    """ONNX name -> the tensor inside the converted torch model, matched by content.

    onnx2torch drops the ONNX names: weights absorbed into an nn.Conv2d become `Conv/7.weight`, and
    everything else lands in `initializers.onnx_initializer_<k>` numbered in visit order, where the
    absorbed ones are skipped -- so the index cannot be predicted from the graph. Matching on the
    bytes is exact for weight tensors, and any collision or miss is raised rather than guessed at,
    because a silently mismatched name would write a weight file that loads into the wrong tensor.
    """
    pool = {}
    for n, t in list(torch_model.named_buffers()) + list(torch_model.named_parameters()):
        pool.setdefault(_digest(t.detach().numpy()), []).append((n, t))

    out = {}
    for name in wanted:
        cand = pool.get(_digest(init[name]), [])
        if len(cand) != 1:
            raise SystemExit(
                "cannot place ONNX tensor '%s' (shape %s) in the converted model: %d matches. "
                "Content matching needs each trained tensor to be byte-unique." %
                (name, init[name].shape, len(cand)))
        out[name] = cand[0][1]
    return out


# ---- the weight file -------------------------------------------------------------------------
def save_weights(path, named):
    """name -> float32 array, in the format pure/train_rec.hpp reads."""
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<Q", len(named)))
        for name, arr in named:
            b = name.encode("utf-8")
            f.write(struct.pack("<Q", len(b)))
            f.write(b)
            a = np.ascontiguousarray(arr, dtype=np.float32).ravel()
            f.write(struct.pack("<Q", a.size))
            f.write(a.tobytes())
    return len(named)


def load_weights(path):
    with open(path, "rb") as f:
        if f.read(8)[:7] != MAGIC[:7]:
            raise SystemExit("%s: not a ppocr weight file" % path)
        (n,) = struct.unpack("<Q", f.read(8))
        out = {}
        for _ in range(n):
            (ln,) = struct.unpack("<Q", f.read(8))
            name = f.read(ln).decode("utf-8")
            (nv,) = struct.unpack("<Q", f.read(8))
            out[name] = np.frombuffer(f.read(nv * 4), dtype=np.float32).copy()
    return out


# ---- data ------------------------------------------------------------------------------------
def load_list(path):
    """PaddleOCR's `<image path>\\t<text>`, paths relative to the list file."""
    base = os.path.dirname(os.path.abspath(path))
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if "\t" not in line:
                continue
            p, text = line.split("\t", 1)
            if not text:
                continue
            out.append((p if os.path.isabs(p) else os.path.join(base, p), text))
    return out


class Encoder:
    """Longest-match over the dictionary, because entries can be several bytes or codepoints.

    A character the dictionary cannot emit raises. Dropping it instead would teach the model to
    suppress a character it is supposed to produce, and the loss falls either way.
    """

    def __init__(self, table):
        self.by_text = {}
        self.longest = 1
        for i, s in enumerate(table):
            if i and s and s not in self.by_text:
                self.by_text[s] = i
                self.longest = max(self.longest, len(s))

    def encode(self, s):
        ids, i = [], 0
        while i < len(s):
            for n in range(min(self.longest, len(s) - i), 0, -1):
                k = self.by_text.get(s[i:i + n])
                if k is not None:
                    ids.append(k)
                    i += n
                    break
            else:
                raise KeyError(s[i])
        return ids


def forward(model, img_path, img_h):
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    x = torch.from_numpy(rec_input(bgr, img_h))
    return model(x)[0]                                   # [T, C], softmaxed by the graph


@torch.no_grad()
def evaluate(model, data, table, img_h, limit=0):
    seen = hit = 0
    for path, text in (data[:limit] if limit else data):
        probs = forward(model, path, img_h)
        if probs is None:
            continue
        got, _ = ctc_greedy(probs.numpy(), table)
        hit += (got == text)
        seen += 1
    return hit, seen


# ---- the loop --------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--img-h", type=int, default=48)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume")
    ap.add_argument("--out", default="rec_finetuned.bin")
    ap.add_argument("--rec", default=REC)
    ap.add_argument("--dict", default=DICT)
    args = ap.parse_args()

    import onnx2torch                                    # imported late: only training needs it

    model_proto, moved = prepare(args.rec)
    wanted, init = select(model_proto.graph)
    model = onnx2torch.convert(model_proto)
    model.eval()                                         # freeze BatchNorm's running statistics
    params = map_params(model, wanted, init)
    table = load_dict(args.dict)
    print("rec: %d nodes (%d constants folded in), dict %d classes" %
          (len(model_proto.graph.node), moved, len(table)))

    if args.resume:
        got = load_weights(args.resume)
        n = 0
        for name, t in params.items():
            v = got.get(name)
            if v is None:
                continue
            if v.size != t.numel():
                print("  weight '%s' is %d values here but %d in the file, skipped" %
                      (name, t.numel(), v.size))
                continue
            with torch.no_grad():
                t.copy_(torch.from_numpy(v).view_as(t))
            n += 1
        print("resumed %d weights from %s" % (n, args.resume))

    for t in params.values():
        t.requires_grad_(True)
    plist = list(params.values())
    total = sum(t.numel() for t in plist)

    train = load_list(args.data)
    val = load_list(args.val) if args.val else []
    print("data: %d train, %d val" % (len(train), len(val)))
    if val:
        hit, seen = evaluate(model, val, table, args.img_h, 200)
        print("before: val exact %.1f%% (%d/%d)" % (100.0 * hit / max(1, seen), hit, seen))

    print("training %d tensors (%.2f M values) on %d samples" % (len(plist), total / 1e6, len(train)))

    enc = Encoder(table)
    ctc = torch.nn.CTCLoss(blank=0, reduction="sum", zero_infinity=True)
    opt = torch.optim.Adam(plist, lr=args.lr)
    rng = random.Random(args.seed)
    order = list(range(len(train)))
    rng.shuffle(order)
    cursor = 0
    run_loss, run_n = 0.0, 0

    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        used, batch_loss = 0, 0.0
        for _ in range(args.batch):
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            path, text = train[order[cursor]]
            cursor += 1
            try:
                ids = enc.encode(text)
            except KeyError as e:
                print("  skipping %s: '%s' is not in the dictionary" % (path, e.args[0]))
                continue
            probs = forward(model, path, args.img_h)
            if probs is None or probs.shape[0] < len(ids):
                continue                                  # target longer than the frames available
            lp = torch.log(probs.clamp_min(1e-7)).unsqueeze(1)          # [T, N=1, C]
            loss = ctc(lp, torch.tensor([ids], dtype=torch.long),
                       torch.tensor([probs.shape[0]]), torch.tensor([len(ids)]))
            loss.backward()
            batch_loss += float(loss.detach())
            used += 1
        if not used:
            continue
        for t in plist:                                   # gradients were summed over the batch
            if t.grad is not None:
                t.grad /= used
        torch.nn.utils.clip_grad_norm_(plist, args.clip)
        opt.step()
        run_loss += batch_loss / used
        run_n += 1

        last = step + 1 == args.steps
        if args.eval_every > 0 and ((step + 1) % args.eval_every == 0 or last):
            line = "  step %5d  loss %8.4f" % (step + 1, run_loss / max(1, run_n))
            run_loss, run_n = 0.0, 0
            if val:
                hit, seen = evaluate(model, val, table, args.img_h, 200)
                line += "   val exact %.1f%% (%d/%d)" % (100.0 * hit / max(1, seen), hit, seen)
            print(line, flush=True)

    if val:
        hit, seen = evaluate(model, val, table, args.img_h, 200)
        print("after:  val exact %.1f%% (%d/%d)" % (100.0 * hit / max(1, seen), hit, seen))

    named = [(name, t.detach().numpy()) for name, t in params.items()]
    print("wrote %s (%d tensors)" % (args.out, save_weights(args.out, named)))


if __name__ == "__main__":
    main()
