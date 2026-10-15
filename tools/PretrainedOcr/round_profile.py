"""Registry of research rounds: owned roots, byte quotas and immutable predecessors.

A round profile is the single source for where a research round may write, how
many new bytes it may create and which closed GPU ledger it binds. Lookups are
root-scoped: a sibling directory sharing a name prefix never inherits a quota.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO / 'artifacts'
GIB = 1024 ** 3
DEFAULT_SPECIALIST_FAMILIES = 2


@dataclass(frozen=True)
class RoundProfile:
    name: str
    change: str
    root: Path
    maximum_bytes: int
    owned_bytes: int
    previous_budget: Path
    previous_sha256: str
    previous_used_seconds: float
    maximum_seconds: float = 7200.0
    specialist_families: int = DEFAULT_SPECIALIST_FAMILIES

    def __post_init__(self):
        if type(self.maximum_bytes) is not int or type(self.owned_bytes) is not int or not 0 < self.owned_bytes <= self.maximum_bytes:
            raise ValueError('Round quotas must be positive integers with owned bytes inside the round quota')
        if len(self.previous_sha256) != 64 or any(c not in '0123456789abcdef' for c in self.previous_sha256):
            raise ValueError('Round predecessor needs a lowercase SHA256')
        if Path(self.previous_budget).resolve().is_relative_to(Path(self.root).resolve()):
            raise ValueError('Round predecessor ledger must stay outside the round root')
        if type(self.specialist_families) is not int or not 1 <= self.specialist_families <= 2:
            raise ValueError('A round freezes one or two specialist families')
        seconds = self.maximum_seconds
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 0 <= seconds <= 7200:
            raise ValueError('A round authorizes between 0 and 7200 GPU seconds; 0 means a CPU-only round')


QUALITY = RoundProfile(
    name='quality-optimization-20261012', change='improve-region-kaomoji-quality',
    root=ARTIFACTS / 'quality-optimization-20261012', maximum_bytes=GIB, owned_bytes=350 * 1024 ** 2,
    previous_budget=ARTIFACTS / 'pretrained-round-20261011/gpu-budget.json',
    previous_sha256='6d4aa4827bf41f84f99a49610a17611dd07185351c6f7a749da8881d250b8497',
    previous_used_seconds=7011.043364, specialist_families=2)
GRAMMAR = RoundProfile(
    name='kaomoji-grammar-20261013', change='improve-kaomoji-grammar-routing-fusion',
    root=ARTIFACTS / 'kaomoji-grammar-20261013', maximum_bytes=2 * GIB, owned_bytes=900 * 1024 ** 2,
    previous_budget=ARTIFACTS / 'quality-optimization-20261012/gpu-budget.json',
    previous_sha256='b7d8f5186d7c19ca6464ea07d72e19eb31146b0a8a1a80418d6bc5f3d10174b3',
    previous_used_seconds=1266.4519999999939, specialist_families=1)
# CPU-only candidate-localization round: a zero-second ledger bound to the closed grammar ledger,
# so no GPU lease can open while freezes and first-use receipts keep their ledger binding.
CANDIDATES = RoundProfile(
    name='region-candidates-20261013', change='improve-region-candidate-localization',
    root=ARTIFACTS / 'region-candidates-20261013', maximum_bytes=GIB, owned_bytes=600 * 1024 ** 2,
    previous_budget=ARTIFACTS / 'kaomoji-grammar-20261013/gpu-budget.json',
    previous_sha256='b2feaecb0b24dadfd697636899df545a457c89a2ab3da09d2ebb16f4ac67dde0',
    previous_used_seconds=3252.876000000001, maximum_seconds=0.0, specialist_families=1)
# CPU-only real-rendering evidence round: bound to the closed zero-second candidates ledger, so the
# predecessor chain stays unbroken while no GPU lease can open.
REAL_RENDERING = RoundProfile(
    name='real-rendering-20261015', change='improve-real-rendering-bracket-evidence',
    root=ARTIFACTS / 'real-rendering-20261015', maximum_bytes=GIB, owned_bytes=600 * 1024 ** 2,
    previous_budget=ARTIFACTS / 'region-candidates-20261013/gpu-budget.json',
    previous_sha256='a650a5777d347fd15efce32e32a2c4e36515031345b8da35e4224c8eb9161e3d',
    previous_used_seconds=0.0, maximum_seconds=0.0, specialist_families=1)
# CPU-only measurement round on browser real-font renderings: bound to the closed zero-second
# real-rendering ledger; no rule, configuration or model changes.
BROWSER_RENDERING = RoundProfile(
    name='browser-rendering-20261015', change='add-browser-rendering-evaluation',
    root=ARTIFACTS / 'browser-rendering-20261015', maximum_bytes=GIB, owned_bytes=600 * 1024 ** 2,
    previous_budget=ARTIFACTS / 'real-rendering-20261015/gpu-budget.json',
    previous_sha256='29bfea13764ff5952e9627ef3c404fa51858996d179a4db38bdd97beb0db38ca',
    previous_used_seconds=0.0, maximum_seconds=0.0, specialist_families=1)
ROUNDS = (QUALITY, GRAMMAR, CANDIDATES, REAL_RENDERING, BROWSER_RENDERING)
CURRENT = BROWSER_RENDERING


def _resolved(path):
    return Path(path).resolve()


def profile_for(path, rounds=None):
    """Return the profile owning ``path`` or None; prefixes never match."""
    target = _resolved(path)
    for profile in (ROUNDS if rounds is None else rounds):
        root = _resolved(profile.root)
        if target == root or target.is_relative_to(root):
            return profile
    return None


def round_root(path, default=None):
    profile = profile_for(path)
    return profile.root if profile is not None else default


def specialist_families(path):
    profile = profile_for(path)
    return profile.specialist_families if profile is not None else DEFAULT_SPECIALIST_FAMILIES


def by_name(name):
    for profile in ROUNDS:
        if profile.name == name:
            return profile
    raise KeyError('Unknown research round: ' + str(name))
