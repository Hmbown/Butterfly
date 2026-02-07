import torch

from hcsa.cycles import (
    OnlineInsertionState,
    cycle_prev_next_from_perm,
    greedy_cycle,
    online_insertion_step,
    random_cycle,
    validate_cycle_perm,
)


def test_random_cycle_is_permutation():
    g = torch.Generator(device="cpu").manual_seed(0)
    perm = random_cycle(17, generator=g, device=torch.device("cpu"))
    validate_cycle_perm(perm, 17)


def test_prev_next_consistent():
    perm = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    prev, nxt = cycle_prev_next_from_perm(perm)
    # successor of 2 should be 0 (since 2->0)
    assert int(nxt[2]) == 0
    # predecessor of 0 should be 2
    assert int(prev[0]) == 2
    # wrap-around: successor of 1 should be 2
    assert int(nxt[1]) == 2
    assert int(prev[2]) == 1


def test_greedy_cycle_is_permutation():
    torch.manual_seed(0)
    T, d = 32, 16
    r = torch.randn(T, d)
    perm = greedy_cycle(r, start=0)
    validate_cycle_perm(perm, T)


def test_online_insertion_step_valid():
    torch.manual_seed(0)
    # build for T=8 using a prior cycle of T-1 nodes
    T, d = 8, 8
    r = torch.randn(T, d)
    base = torch.randperm(T - 1)
    state = OnlineInsertionState(perm=base)
    state2 = online_insertion_step(state, r)
    validate_cycle_perm(state2.perm, T)
