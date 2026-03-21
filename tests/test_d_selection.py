"""Tests for principled d (num_cycles) selection utilities."""

from bna.cycles import max_edge_disjoint_cycles, recommended_num_cycles


def test_recommended_num_cycles_1024():
    d = recommended_num_cycles(T=1024)
    assert 11 <= d <= 30, f"Expected d in [11,30] for T=1024, got {d}"


def test_recommended_num_cycles_2():
    assert recommended_num_cycles(T=2) == 2


def test_max_edge_disjoint_cycles_100():
    assert max_edge_disjoint_cycles(T=100) == 49


def test_max_edge_disjoint_cycles_3():
    assert max_edge_disjoint_cycles(T=3) == 1


def test_max_edge_disjoint_cycles_2():
    assert max_edge_disjoint_cycles(T=2) == 0
