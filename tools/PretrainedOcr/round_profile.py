"""Registry of research rounds: owned roots, byte quotas and immutable predecessors.

A round profile is the single source for where a research round may write, how
many new bytes it may create and which closed GPU ledger it binds. Lookups are
root-scoped: a sibling directory sharing a name prefix never inherits a quota.
"""
from __future__ import annotations

from dataclasses import dataclass
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
ROUNDS = (QUALITY, GRAMMAR)
CURRENT = GRAMMAR


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
