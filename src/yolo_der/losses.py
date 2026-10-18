"""Entropy and load-balancing regularisers for the routing weights (E1, spec 7.6).

The paper calls DER "entropy-aware" (Sec. 3.4) but never measures or optimises any
entropy. Extension E1 closes that gap from both ends: ``analysis/entropy.py`` *measures*
routing entropy on a trained model, and these two terms let us *optimise* it and see
whether a sharper router actually detects better.

Both terms are 0 unless the config enables them, so the main ablation is unaffected.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from yolo_der.models.common import RecordingMixin

EPS = 1e-8

#: Which modules the E1 regularisers may act on.
VALID_TARGETS = frozenset({"all", "der", "enhancenet"})


def routing_entropy(p: torch.Tensor) -> torch.Tensor:
    """Mean Shannon entropy of a batch of routing distributions.

    Args:
        p: ``(B, N)`` rows that each sum to 1 -- DER's ``o_hat`` or EnhanceNet's ``beta``.

    Returns:
        A scalar: ``mean_b [ -sum_n p log(p + eps) ]``, in nats.

    Low entropy means the router commits to a path; entropy near ``log N`` means it is
    just averaging the paths, which would make the whole "dynamic routing" story a
    fixed-weight ensemble. That is the question E1 asks.
    """
    return (-(p * torch.log(p + EPS)).sum(dim=1)).mean()


def normalised_routing_entropy(p: torch.Tensor) -> torch.Tensor:
    """:func:`routing_entropy` divided by ``log N``, so 0 is decisive and 1 is uniform."""
    n = p.shape[1]
    if n < 2:
        return torch.zeros((), dtype=p.dtype, device=p.device)
    return routing_entropy(p) / math.log(n)


def load_balance(p: torch.Tensor) -> torch.Tensor:
    """KL divergence from the batch-mean routing distribution to the uniform one.

    ``KL(mean_b p || uniform) = sum_n q_n log(N q_n)`` with ``q = mean_b p``.

    This is the Mixture-of-Experts trick: :func:`routing_entropy` pushes each *image*
    towards one path, which on its own would happily send *every* image down the same
    path. This term penalises that collapse by asking the paths to be used equally
    *across* the batch. The two are complementary, not contradictory -- confident per
    image, balanced in aggregate.
    """
    n = p.shape[1]
    q = p.mean(dim=0)
    return (q * torch.log(n * q + EPS)).sum()


def collect_routing(model: nn.Module, target: str = "all") -> list[torch.Tensor]:
    """Gather the live routing weights that recording modules stashed this forward.

    Modules only populate ``routing_for_loss`` when built with ``collect_for_loss=True``,
    and they clear it at the start of each forward, so a stale graph is never reused.

    Args:
        model: the model whose modules were just run forward.
        target: ``"all"``, ``"der"`` or ``"enhancenet"`` -- which modules the regularisers
            act on.

    Why ``target`` exists: the first entropy sweep pushed **every** routing distribution at
    once, and EnhanceNet's branch gate collapsed alongside DER's router (normalised entropy
    0.80 -> 0.47 -> 0.005 as the weight went 0.1 -> 1.0 -> 5.0). The accompanying -0.15
    mAP50 therefore could not be attributed to decisive routing: it may have been the loss
    of a gate that was doing useful work. Targeting one module at a time is what makes the
    claim attributable.
    """
    from yolo_der.models.der import DER

    # Raise rather than fall through. A typo in `entropy_target` would otherwise silently
    # push every module -- i.e. quietly run a different experiment from the one named in
    # the config, which is the failure this whole targeting change exists to prevent.
    if target not in VALID_TARGETS:
        raise ValueError(f"entropy_target must be one of {sorted(VALID_TARGETS)}, got {target!r}")

    out: list[torch.Tensor] = []
    for module in model.modules():
        if not isinstance(module, RecordingMixin) or module.routing_for_loss is None:
            continue
        is_der = isinstance(module, DER)
        if target == "der" and not is_der:
            continue
        if target == "enhancenet" and is_der:
            continue
        out.append(module.routing_for_loss)
    return out


def regularisers(
    model: nn.Module,
    *,
    entropy_weight: float = 0.0,
    load_balance_weight: float = 0.0,
    target: str = "all",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return ``(total, breakdown)`` for the enabled routing regularisers.

    Args:
        model: the model whose modules were just run forward.
        entropy_weight: weight on mean routing entropy. **Positive sharpens routing**,
            because the term is added to the loss and entropy is minimised. Default 0.01
            in ``der_enh_entreg``.
        load_balance_weight: weight on the collapse penalty. Default 0.01.
        target: which modules to push -- ``"all"``, ``"der"`` or ``"enhancenet"``.

    Returns:
        A scalar tensor (0 and detached when both weights are 0, so the baseline path
        allocates nothing), plus a dict of plain floats for logging.
    """
    device = next(model.parameters()).device
    zero = torch.zeros((), device=device)
    if entropy_weight == 0.0 and load_balance_weight == 0.0:
        return zero, {}

    weights = collect_routing(model, target)
    if not weights:
        return zero, {}

    # Mean over insertion points: the number of modules must not change the term's scale.
    ent = torch.stack([routing_entropy(p) for p in weights]).mean()
    bal = torch.stack([load_balance(p) for p in weights]).mean()
    total = entropy_weight * ent + load_balance_weight * bal

    return total, {
        "routing_entropy": float(ent.detach()),
        "routing_load_balance": float(bal.detach()),
        "routing_reg_total": float(total.detach()),
    }
