"""B6: faithful (function-preserving) neuron permutation on a full model.

Real NeuPerm permutes a layer's output channels and the matching input channels of
the consuming layer, leaving the network's function unchanged. We implement this
across a whole sequential conv net (conv -> BN -> ReLU -> ... -> AvgPool -> Linear),
permuting each conv's out-channels together with its BN params and the next layer's
in-channels. Pool/ReLU are channel-wise, so the permutation passes through exactly.

`permute_smallcnn` mutates a copy in place and is verified to keep logits identical
(max|Δ| ~ 1e-6, float rounding only). Use it to test which payloads survive a
*faithful* full-model permutation: flat/positional payloads die; payloads indexed
by the permutation-invariant SPATIAL position (kh,kw) survive.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def _conv_bn_chain(net):
    """Return (convs, bns) in forward order for a SmallCNN-style net.f of blocks."""
    convs, bns = [], []
    for m in net.f:
        if isinstance(m, nn.Sequential):        # a conv-bn-relu block
            convs.append(m[0]); bns.append(m[1])
    return convs, bns


@torch.no_grad()
def permute_smallcnn(net, seed: int = 0):
    """Apply a function-preserving neuron permutation to every conv layer of a
    SmallCNN. net.h is the final Linear consumer of the last conv. Mutates net."""
    g = torch.Generator().manual_seed(seed)
    convs, bns = _conv_bn_chain(net)
    for k, (c, b) in enumerate(zip(convs, bns)):
        O = c.weight.shape[0]
        perm = torch.randperm(O, generator=g)
        c.weight.copy_(c.weight[perm])
        if c.bias is not None:
            c.bias.copy_(c.bias[perm])
        # BatchNorm params follow the same channel permutation
        b.weight.copy_(b.weight[perm]); b.bias.copy_(b.bias[perm])
        b.running_mean.copy_(b.running_mean[perm]); b.running_var.copy_(b.running_var[perm])
        # permute the consumer's input channels to cancel it out
        if k + 1 < len(convs):
            nxt = convs[k + 1]
            nxt.weight.copy_(nxt.weight[:, perm])
        else:
            net.h.weight.copy_(net.h.weight[:, perm])   # AvgPool->flatten keeps channel order
    return net


@torch.no_grad()
def max_logit_diff(net_a, net_b, x):
    """Max |logit| difference between two nets on input x (0 => function-identical)."""
    return float((net_a(x) - net_b(x)).abs().max().item())


# --- B6 at real scale: faithful NeuPerm on a torchvision ResNet-18 (BasicBlock) ---
#
# A residual block out = relu(bn2(conv2(relu(bn1(conv1(x))))) + identity) constrains
# the permutation: conv2's OUTPUT channels and the identity stream must share ONE
# permutation, else the add breaks. So two independent permutation families exist:
#   (1) INNER (mid) channels of each block  -- conv1.out <-> conv2.in (+ bn1): always free.
#   (2) STREAM channels per stage           -- conv1.in, conv2.out/bn2, downsample,
#       and the fc input, all tied to one per-stage permutation (stem+layer1 share it).
# Both are function-preserving; we apply both (verify with max_logit_diff ~ 0).

def _perm_out(conv, perm):
    conv.weight.copy_(conv.weight[perm])
    if conv.bias is not None:
        conv.bias.copy_(conv.bias[perm])


def _perm_in(conv, perm):
    conv.weight.copy_(conv.weight[:, perm])


def _perm_bn(bn, perm):
    bn.weight.copy_(bn.weight[perm]); bn.bias.copy_(bn.bias[perm])
    bn.running_mean.copy_(bn.running_mean[perm]); bn.running_var.copy_(bn.running_var[perm])


@torch.no_grad()
def permute_resnet18(model, seed: int = 0):
    """Function-preserving neuron permutation of a torchvision ResNet-18. Mutates the
    model in place; logits are unchanged (verify with max_logit_diff)."""
    g = torch.Generator().manual_seed(seed)
    stages = [model.layer1, model.layer2, model.layer3, model.layer4]

    # (1) inner-channel permutation per block (the bottleneck/mid dimension)
    for st in stages:
        for blk in st:
            mid = blk.conv1.weight.shape[0]
            p = torch.randperm(mid, generator=g)
            _perm_out(blk.conv1, p); _perm_bn(blk.bn1, p); _perm_in(blk.conv2, p)

    # (2) stream-channel permutation per stage (stem + layer1 share perms[0])
    widths = [st[0].conv2.weight.shape[0] for st in stages]   # 64,128,256,512
    perms = [torch.randperm(w, generator=g) for w in widths]
    _perm_out(model.conv1, perms[0]); _perm_bn(model.bn1, perms[0])
    prev = perms[0]                                           # stream entering current stage
    for si, st in enumerate(stages):
        P = perms[si]
        for bi, blk in enumerate(st):
            _perm_in(blk.conv1, prev if bi == 0 else P)       # conv1 reads the incoming stream
            _perm_out(blk.conv2, P); _perm_bn(blk.bn2, P)     # conv2 writes this stage's stream
            if blk.downsample is not None:                    # 1x1 shortcut: prev -> P
                _perm_in(blk.downsample[0], prev)
                _perm_out(blk.downsample[0], P); _perm_bn(blk.downsample[1], P)
        prev = P
    model.fc.weight.copy_(model.fc.weight[:, perms[-1]])      # fc reads the final stream
    return model
