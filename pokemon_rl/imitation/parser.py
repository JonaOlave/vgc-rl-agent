"""
Parse Pokemon Showdown replay logs into (observation, action) pairs.

observation : np.ndarray (925,)  float32 — VGC embedding (same as VGCEnv)
action      : np.ndarray (2,)    int64   — [action_slot0, action_slot1]

Action encoding (per slot, same as DoublesEnv):
  0          → pass
  1–6        → switch to team slot N
  7–11       → move 1 to target (-2,-1,0,1,2)
  12–16      → move 2  …
  17–21      → move 3  …
  22–26      → move 4  …
  27–46      → moves 1-4 + mega
  47–66      → moves 1-4 + z-move
  67–86      → moves 1-4 + dynamax
  87–106     → moves 1-4 + terastallize

PS message format note
----------------------
Lines in a PS replay log start with ``|``, e.g. ``|move|p1a: Foo|Tackle|p2a: Bar``.
``line.split("|")`` gives ``["", "move", "p1a: Foo", ...]`` — the leading empty
string is required by ``DoubleBattle.parse_message``, which checks ``event[1]``
for the message type.  Manual parsing uses ``line[1:].split("|")`` (index 0 =
type), while ``parse_message`` calls use the full split (index 1 = type).
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from poke_env.battle import Move
from poke_env.battle.double_battle import DoubleBattle

from pokemon_rl.vgc.embedding import embed_battle as _embed

_LOG = logging.getLogger(__name__)

# move_target for p1 given the target position string from the log.
# DoubleBattle constants: POKEMON_1_POSITION=-1, POKEMON_2_POSITION=-2,
# OPPONENT_1_POSITION=1, OPPONENT_2_POSITION=2, EMPTY_TARGET_POSITION=0
_TARGET_FOR_P1: Dict[str, int] = {
    "p1a": DoubleBattle.POKEMON_1_POSITION,   # -1
    "p1b": DoubleBattle.POKEMON_2_POSITION,   # -2
    "p2a": DoubleBattle.OPPONENT_1_POSITION,  #  1
    "p2b": DoubleBattle.OPPONENT_2_POSITION,  #  2
}


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _manual_parts(line: str) -> List[str]:
    """Split ``|type|a|b|`` → ``["type", "a", "b", ""]``.  Index 0 = message type."""
    return line[1:].split("|")


def _battle_parts(line: str) -> List[str]:
    """Split ``|type|a|b|`` → ``["", "type", "a", "b", ""]``.  For parse_message."""
    return line.split("|")


def _feed(battle: DoubleBattle, line: str) -> None:
    """Feed one log line to the battle object, silently ignoring errors."""
    try:
        battle.parse_message(_battle_parts(line))
    except Exception:
        pass


def _get_players(lines: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Return (p1_username, p2_username) from log lines."""
    p1 = p2 = None
    for line in lines:
        if not line.startswith("|player|"):
            continue
        parts = _manual_parts(line)  # ["player", "p1", "username", ...]
        if len(parts) < 3:
            continue
        role, username = parts[1], parts[2]
        if role == "p1":
            p1 = username
        elif role == "p2":
            p2 = username
        if p1 and p2:
            break
    return p1, p2


# ─────────────────────────────────────────────────────────────────────────────
# Action encoding
# ─────────────────────────────────────────────────────────────────────────────

def _encode_move_action(
    move_name: str,
    target_pos: Optional[str],
    battle: DoubleBattle,
    pos: int,
    terastallize: bool = False,
    mega: bool = False,
    z_move: bool = False,
    dynamax: bool = False,
) -> Optional[int]:
    """Encode a move into an action integer [7, 106]."""
    active = (battle.active_pokemon or [None, None])[pos]
    if active is None:
        return None

    move_id = Move.retrieve_id(move_name)
    known = list(active.moves.values())[:4]
    known_ids = [m.id for m in known]

    if move_id in known_ids:
        move_idx = known_ids.index(move_id)
    else:
        # Move not yet registered; check available_moves
        avail = battle.available_moves[pos] if len(battle.available_moves) > pos else []
        avail_ids = [m.id for m in avail]
        if len(avail_ids) == 1:
            move_idx = 0  # Struggle or locked move
        elif move_id in avail_ids and len(known) < 4:
            move_idx = len(known)  # First use, appears after known
        else:
            return None

    target_int = _TARGET_FOR_P1.get(
        target_pos or "", DoubleBattle.EMPTY_TARGET_POSITION
    )

    if terastallize:
        gimmick = 4
    elif mega:
        gimmick = 1
    elif z_move:
        gimmick = 2
    elif dynamax:
        gimmick = 3
    else:
        gimmick = 0

    encoded_target = target_int + 2  # [-2,2] → [0,4]
    return 7 + 5 * move_idx + encoded_target + 20 * gimmick


def _encode_switch_action(species_raw: str, battle: DoubleBattle) -> Optional[int]:
    """Encode a switch-in into an action integer [1, 6]."""
    # species_raw: "Urshifu-*, L50, M" — take name before first comma
    species = species_raw.split(",")[0].strip()
    species_lower = species.lower()

    team = list(battle.team.values())
    for i, pkmn in enumerate(team):
        if pkmn.species.lower() == species_lower:
            return i + 1
        if pkmn.base_species.lower() == species_lower:
            return i + 1

    # Fallback: prefix match for forms like "Urshifu-*"
    base = re.split(r"[-*]", species_lower)[0].strip()
    for i, pkmn in enumerate(team):
        if pkmn.base_species.lower().startswith(base):
            return i + 1

    return None


def _extract_actions(
    turn_lines: List[str],
    battle: DoubleBattle,
) -> Optional[np.ndarray]:
    """
    Scan the turn's lines and return [action_slot0, action_slot1] for p1.
    Only captures VOLUNTARY actions (before any faint mid-turn).
    Returns None if no p1 action was found.
    """
    actions = [0, 0]
    found = [False, False]

    tera: set = set()
    mega: set = set()
    zmov: set = set()
    dmax: set = set()
    p1_fainted: set = set()

    for line in turn_lines:
        if not line.startswith("|"):
            continue
        parts = _manual_parts(line)  # parts[0] = msg type
        if not parts:
            continue
        msg = parts[0]

        if msg == "-faint" and len(parts) >= 2:
            slot = parts[1][:3]
            if slot == "p1a":
                p1_fainted.add(0)
            elif slot == "p1b":
                p1_fainted.add(1)

        elif msg == "-terastallize" and len(parts) >= 2:
            slot = parts[1][:3]
            if slot == "p1a":
                tera.add(0)
            elif slot == "p1b":
                tera.add(1)

        elif msg == "-mega" and len(parts) >= 2:
            slot = parts[1][:3]
            if slot == "p1a":
                mega.add(0)
            elif slot == "p1b":
                mega.add(1)

        elif msg == "-zpower" and len(parts) >= 2:
            slot = parts[1][:3]
            if slot == "p1a":
                zmov.add(0)
            elif slot == "p1b":
                zmov.add(1)

        elif msg == "-dynamax" and len(parts) >= 2:
            slot = parts[1][:3]
            if slot == "p1a":
                dmax.add(0)
            elif slot == "p1b":
                dmax.add(1)

        elif msg == "move" and len(parts) >= 3:
            actor = parts[1][:3]
            if actor not in ("p1a", "p1b"):
                continue
            pos = 0 if actor == "p1a" else 1
            if found[pos]:
                continue

            move_name = parts[2]
            target_raw = parts[3] if len(parts) >= 4 else ""
            # target_raw might be "p2a: Nickname" — keep only position prefix
            target_pos = target_raw[:3] if len(target_raw) >= 3 else None

            action = _encode_move_action(
                move_name=move_name,
                target_pos=target_pos,
                battle=battle,
                pos=pos,
                terastallize=(pos in tera),
                mega=(pos in mega),
                z_move=(pos in zmov),
                dynamax=(pos in dmax),
            )
            if action is not None:
                actions[pos] = action
                found[pos] = True

        elif msg == "switch" and len(parts) >= 3:
            actor = parts[1][:3]
            if actor not in ("p1a", "p1b"):
                continue
            pos = 0 if actor == "p1a" else 1
            if found[pos]:
                continue
            # Forced switch (after faint) is not a voluntary decision
            if pos in p1_fainted:
                continue

            action = _encode_switch_action(parts[2], battle)
            if action is not None:
                actions[pos] = action
                found[pos] = True

    if not any(found):
        return None

    return np.array(actions, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def parse_replay(
    replay_data: dict,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Parse a replay JSON dict into (observation, action) training pairs.

    Parameters
    ----------
    replay_data : dict
        Replay JSON as returned by the PS API (must have a ``"log"`` key).

    Returns
    -------
    List of (obs, action) tuples where
        obs    : np.ndarray (925,) float32
        action : np.ndarray (2,)   int64
    """
    log: str = replay_data.get("log", "")
    replay_id: str = replay_data.get("id", "unknown")

    lines = [ln for ln in log.split("\n") if ln.startswith("|")]
    if not lines:
        return []

    p1_username, _ = _get_players(lines)
    if not p1_username:
        _LOG.debug("No p1 username in replay %s", replay_id)
        return []

    battle = DoubleBattle(
        battle_tag=replay_id,
        username=p1_username,
        logger=_LOG,
        gen=9,
    )

    # ── Split into: preamble (before |turn|) and per-turn segments ──
    preamble: List[str] = []
    segments: List[Tuple[int, List[str]]] = []
    cur_turn: Optional[int] = None
    cur_lines: List[str] = []

    for line in lines:
        parts = _manual_parts(line)
        if not parts:
            continue
        if parts[0] == "turn":
            if cur_turn is not None:
                segments.append((cur_turn, cur_lines))
            try:
                cur_turn = int(parts[1])
            except (ValueError, IndexError):
                cur_turn = 0
            cur_lines = []
        elif cur_turn is None:
            preamble.append(line)
        else:
            cur_lines.append(line)

    if cur_turn is not None:
        segments.append((cur_turn, cur_lines))

    # ── Process preamble to initialise battle state ──
    for line in preamble:
        _feed(battle, line)

    # ── Process each turn ──
    samples: List[Tuple[np.ndarray, np.ndarray]] = []

    for turn_n, turn_lines in segments:
        # Advance to new turn marker
        _feed(battle, f"|turn|{turn_n}")

        # Capture observation BEFORE turn effects (correct RL convention)
        try:
            obs = _embed(battle)
        except Exception as e:
            _LOG.debug("embed_battle failed turn %d replay %s: %s", turn_n, replay_id, e)
            obs = None

        # Process turn lines FIRST — this registers newly-used moves into
        # pokemon.moves, which _extract_actions needs for move-index lookup.
        # Effects (HP changes, status, etc.) also update, but that's OK because
        # the observation was already snapped above.
        for line in turn_lines:
            _feed(battle, line)

        # Extract actions from turn lines using updated (post-turn) battle state
        action: Optional[np.ndarray] = None
        if obs is not None:
            try:
                action = _extract_actions(turn_lines, battle)
            except Exception as e:
                _LOG.debug("action extraction failed turn %d: %s", turn_n, e)

        if obs is not None and action is not None:
            samples.append((obs, action))

    return samples
