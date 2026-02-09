from __future__ import annotations

from hcsa.discover import get_target, list_targets, resolve_targets


def test_discover_targets_registry_has_expected_ids() -> None:
    targets = list_targets()
    assert set(targets.keys()) == {"k1", "k2", "k3", "k4", "k5"}


def test_resolve_targets_all_returns_five_targets() -> None:
    targets = resolve_targets(["all"])
    assert len(targets) == 5
    assert {t.id for t in targets} == {"k1", "k2", "k3", "k4", "k5"}


def test_get_target_supports_kernel_name_alias() -> None:
    spec = get_target("hcsa_active_row_fused")
    assert spec.id == "k4"
    assert spec.discover_target == "hcsa_active_row"
