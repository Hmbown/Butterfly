from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence, cast

ButterflyPartnerRule = Literal["xor", "bit_reversal", "benes"]
RoleName = Literal["self", "local", "partner", "sink"]

ROLE_ORDER: tuple[RoleName, ...] = ("self", "local", "partner", "sink")
_VALID_PARTNER_RULES: set[str] = {"xor", "bit_reversal", "benes"}


@dataclass(frozen=True)
class NeighborSpec:
    neighbor: int
    roles: tuple[RoleName, ...]


@dataclass(frozen=True)
class ButterflyLayoutMetadata:
    num_blocks: int
    width: int
    stage_idx: int
    stage_count: int
    partner_bits: tuple[int, ...]
    sink_blocks: tuple[int, ...]
    local_window_blocks: int
    partner_count: int
    partner_rule: ButterflyPartnerRule


def butterfly_width(num_blocks: int) -> int:
    if int(num_blocks) <= 1:
        return 1
    return int(math.ceil(math.log2(int(num_blocks))))


def bit_reverse(value: int, width: int) -> int:
    out = 0
    for bit_idx in range(int(width)):
        out = (out << 1) | ((int(value) >> bit_idx) & 1)
    return out


def butterfly_stage_meta(
    *,
    num_blocks: int,
    layer_idx: int,
    partner_rule: ButterflyPartnerRule | str,
) -> tuple[int, int]:
    rule = _normalize_partner_rule(partner_rule)
    width = butterfly_width(num_blocks)
    if rule == "benes":
        stage_count = max(1, (2 * width) - 2)
    else:
        stage_count = width
    stage_idx = int(layer_idx) % int(stage_count)
    return int(stage_idx), int(stage_count)


def butterfly_stage_count(num_blocks: int, partner_rule: ButterflyPartnerRule | str) -> int:
    _stage_idx, stage_count = butterfly_stage_meta(
        num_blocks=int(num_blocks),
        layer_idx=0,
        partner_rule=partner_rule,
    )
    return int(stage_count)


def butterfly_partner_bits(
    *,
    stage_idx: int,
    stage_count: int,
    width: int,
    partner_count: int,
    partner_rule: ButterflyPartnerRule | str,
) -> list[int]:
    rule = _normalize_partner_rule(partner_rule)
    if int(width) <= 0 or int(partner_count) <= 0:
        return []
    if rule == "benes":
        if int(stage_count) <= 1:
            base_phase = 0
        else:
            base_phase = int(stage_idx) % int(stage_count)

        def phase_to_bit(phase: int) -> int:
            if int(width) <= 1:
                return 0
            if int(phase) < int(width):
                return int(phase)
            return int((2 * int(width)) - 2 - int(phase))

        return [
            phase_to_bit((int(base_phase) + int(offset)) % int(stage_count))
            for offset in range(int(partner_count))
        ]
    return [int(stage_idx + offset) % int(width) for offset in range(int(partner_count))]


def butterfly_partner_block(
    *,
    block_idx: int,
    bit_idx: int,
    num_blocks: int,
    partner_rule: ButterflyPartnerRule | str,
    width: int | None = None,
) -> int | None:
    rule = _normalize_partner_rule(partner_rule)
    effective_width = butterfly_width(num_blocks) if width is None else int(width)
    if int(bit_idx) < 0 or int(bit_idx) >= max(1, int(effective_width)):
        return None
    if rule in {"xor", "benes"}:
        partner = int(block_idx) ^ (1 << int(bit_idx))
    else:
        reversed_idx = bit_reverse(int(block_idx), int(effective_width))
        partner = bit_reverse(reversed_idx ^ (1 << int(bit_idx)), int(effective_width))
    if partner < 0 or partner >= int(num_blocks) or partner > int(block_idx):
        return None
    return int(partner)


def butterfly_layout_metadata(
    *,
    num_blocks: int,
    layer_idx: int,
    partner_rule: ButterflyPartnerRule | str,
    partner_count: int,
    sink_count: int,
    local_window_blocks: int,
) -> ButterflyLayoutMetadata:
    rule = _normalize_partner_rule(partner_rule)
    width = butterfly_width(int(num_blocks))
    stage_idx, stage_count = butterfly_stage_meta(
        num_blocks=int(num_blocks),
        layer_idx=int(layer_idx),
        partner_rule=rule,
    )
    partner_bits = tuple(
        butterfly_partner_bits(
            stage_idx=int(stage_idx),
            stage_count=int(stage_count),
            width=int(width),
            partner_count=int(partner_count),
            partner_rule=rule,
        )
    )
    sink_blocks = tuple(range(min(int(sink_count), int(num_blocks))))
    return ButterflyLayoutMetadata(
        num_blocks=int(num_blocks),
        width=int(width),
        stage_idx=int(stage_idx),
        stage_count=int(stage_count),
        partner_bits=partner_bits,
        sink_blocks=sink_blocks,
        local_window_blocks=int(local_window_blocks),
        partner_count=int(partner_count),
        partner_rule=rule,
    )


def build_butterfly_role_map(
    *,
    block_idx: int,
    num_blocks: int,
    local_window_blocks: int,
    sink_blocks: Sequence[int],
    partner_bits: Sequence[int],
    partner_rule: ButterflyPartnerRule | str,
    width: int | None = None,
) -> dict[int, set[RoleName]]:
    role_map: dict[int, set[RoleName]] = {int(block_idx): {"self"}}
    for offset in range(1, int(local_window_blocks) + 1):
        local_block = int(block_idx) - int(offset)
        if local_block < 0:
            break
        role_map.setdefault(int(local_block), set()).add("local")

    for bit_idx in partner_bits:
        partner = butterfly_partner_block(
            block_idx=int(block_idx),
            bit_idx=int(bit_idx),
            num_blocks=int(num_blocks),
            partner_rule=partner_rule,
            width=int(width) if width is not None else None,
        )
        if partner is not None:
            role_map.setdefault(int(partner), set()).add("partner")

    for sink_block in sink_blocks:
        isink = int(sink_block)
        if isink < 0 or isink >= int(num_blocks):
            continue
        role_map.setdefault(isink, set()).add("sink")
    return role_map


def build_butterfly_neighbor_role_row(
    *,
    block_idx: int,
    metadata: ButterflyLayoutMetadata,
) -> list[NeighborSpec]:
    role_map = build_butterfly_role_map(
        block_idx=int(block_idx),
        num_blocks=int(metadata.num_blocks),
        local_window_blocks=int(metadata.local_window_blocks),
        sink_blocks=metadata.sink_blocks,
        partner_bits=metadata.partner_bits,
        partner_rule=metadata.partner_rule,
        width=int(metadata.width),
    )
    ordered_neighbors = _ordered_unique_valid(
        values=(
            [int(block_idx)]
            + [int(block_idx) - int(offset) for offset in range(1, int(metadata.local_window_blocks) + 1)]
            + [
                partner
                for bit_idx in metadata.partner_bits
                for partner in [
                    butterfly_partner_block(
                        block_idx=int(block_idx),
                        bit_idx=int(bit_idx),
                        num_blocks=int(metadata.num_blocks),
                        partner_rule=metadata.partner_rule,
                        width=int(metadata.width),
                    )
                ]
                if partner is not None
            ]
            + [int(sink) for sink in metadata.sink_blocks]
        ),
        upper_bound=int(metadata.num_blocks),
    )
    return [
        NeighborSpec(
            neighbor=int(neighbor),
            roles=ordered_roles(role_map.get(int(neighbor), {"self"})),
        )
        for neighbor in ordered_neighbors
    ]


def ordered_roles(roles: set[str] | set[RoleName]) -> tuple[RoleName, ...]:
    return tuple(role for role in ROLE_ORDER if role in roles)


def _normalize_partner_rule(partner_rule: ButterflyPartnerRule | str) -> ButterflyPartnerRule:
    rule = str(partner_rule)
    if rule not in _VALID_PARTNER_RULES:
        raise ValueError(f"Unsupported Butterfly partner rule: {partner_rule!r}")
    return cast(ButterflyPartnerRule, rule)


def _ordered_unique_valid(values: Sequence[int], *, upper_bound: int) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        ivalue = int(value)
        if ivalue < 0 or ivalue >= int(upper_bound) or ivalue in seen:
            continue
        seen.add(ivalue)
        out.append(ivalue)
    return out


__all__ = [
    "ButterflyLayoutMetadata",
    "ButterflyPartnerRule",
    "NeighborSpec",
    "RoleName",
    "ROLE_ORDER",
    "bit_reverse",
    "build_butterfly_neighbor_role_row",
    "build_butterfly_role_map",
    "butterfly_layout_metadata",
    "butterfly_partner_bits",
    "butterfly_partner_block",
    "butterfly_stage_count",
    "butterfly_stage_meta",
    "butterfly_width",
    "ordered_roles",
]
