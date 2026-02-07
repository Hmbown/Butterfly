"""Property-based tests for cycle construction using Hypothesis.

Every strategy must produce a valid permutation for any T >= 1.
The prev/next arrays must be consistent with the permutation.
"""

from __future__ import annotations

import torch
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from hcsa.cycles import (
    OnlineInsertionState,
    cycle_prev_next_from_perm,
    greedy_cycle,
    online_insertion_step,
    random_cycle,
    validate_cycle_perm,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_valid_cycle(perm: torch.Tensor, T: int) -> None:
    """Check perm is valid and prev/next are self-consistent."""
    validate_cycle_perm(perm, T)
    prev, nxt = cycle_prev_next_from_perm(perm)

    # For every node, following nxt then prev should return to the same node.
    for i in range(T):
        assert int(prev[int(nxt[i])]) == i, f"prev[nxt[{i}]] != {i}"
        assert int(nxt[int(prev[i])]) == i, f"nxt[prev[{i}]] != {i}"

    # Walking the full cycle via nxt should visit every node exactly once.
    visited = set()
    cur = 0
    for _ in range(T):
        assert cur not in visited, f"node {cur} visited twice"
        visited.add(cur)
        cur = int(nxt[cur])
    assert visited == set(range(T))


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@given(T=st.integers(min_value=1, max_value=200))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_random_cycle_valid_for_any_T(T: int) -> None:
    g = torch.Generator(device="cpu").manual_seed(42)
    perm = random_cycle(T, generator=g, device=torch.device("cpu"))
    _assert_valid_cycle(perm, T)


@given(T=st.integers(min_value=2, max_value=64))
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_greedy_cycle_valid_for_any_T(T: int) -> None:
    torch.manual_seed(0)
    r = torch.randn(T, 8)
    perm = greedy_cycle(r, start=0)
    _assert_valid_cycle(perm, T)


@given(T=st.integers(min_value=2, max_value=64))
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_online_insertion_valid_for_any_T(T: int) -> None:
    torch.manual_seed(0)
    r = torch.randn(T, 8)
    base = torch.randperm(T - 1)
    state = OnlineInsertionState(perm=base)
    state2 = online_insertion_step(state, r)
    _assert_valid_cycle(state2.perm, T)


@given(T=st.integers(min_value=1, max_value=200))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_prev_next_inverse(T: int) -> None:
    """prev and nxt must be inverses of each other in a cycle."""
    g = torch.Generator(device="cpu").manual_seed(7)
    perm = random_cycle(T, generator=g, device=torch.device("cpu"))
    prev, nxt = cycle_prev_next_from_perm(perm)
    # nxt[prev[i]] == i for all i
    for i in range(T):
        assert int(nxt[int(prev[i])]) == i
        assert int(prev[int(nxt[i])]) == i


@given(
    T=st.integers(min_value=2, max_value=64),
    start=st.integers(min_value=0, max_value=63),
)
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_greedy_cycle_different_starts(T: int, start: int) -> None:
    start = start % T
    torch.manual_seed(0)
    r = torch.randn(T, 8)
    perm = greedy_cycle(r, start=start)
    _assert_valid_cycle(perm, T)
