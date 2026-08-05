"""
Equipos de ejemplo para VGC (Gen 9).

Para usar un equipo en poke-env, pásalo como string al argumento `team`
del entorno. El formato es el estándar de Pokemon Showdown.

Uso:
    from pokemon_rl.vgc.team import SAMPLE_TEAM_REG_H, SAMPLE_TEAM_CHAMPIONS_REGMB
    env = make_vgc_env(team=SAMPLE_TEAM_CHAMPIONS_REGMB, battle_format="gen9championsvgc2026regmb")
"""

# Equipo Regulation H básico — Koraidon + soporte
# Puedes reemplazarlo con cualquier equipo exportado desde Pokemon Showdown
SAMPLE_TEAM_REG_H = """
Koraidon @ Booster Energy
Ability: Orichalcum Pulse
Level: 50
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Collision Course
- Flare Blitz
- Dragon Claw
- Protect

Flutter Mane @ Choice Specs
Ability: Protosynthesis
Level: 50
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Moonblast
- Shadow Ball
- Mystical Fire
- Dazzling Gleam

Incineroar @ Safety Goggles
Ability: Intimidate
Level: 50
EVs: 252 HP / 4 Atk / 252 SpD
Careful Nature
- Fake Out
- Flare Blitz
- Knock Off
- Parting Shot

Amoonguss @ Rocky Helmet
Ability: Regenerator
Level: 50
EVs: 252 HP / 4 SpA / 252 SpD
Calm Nature
- Spore
- Rage Powder
- Pollen Puff
- Protect

Iron Hands @ Assault Vest
Ability: Quark Drive
Level: 50
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Drain Punch
- Heavy Slam
- Wild Charge
- Fake Out

Palafin-Hero @ Choice Band
Ability: Zero to Hero
Level: 50
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Jet Punch
- Wave Crash
- Close Combat
- Protect
"""

# ─────────────────────────────────────────────────────────────────────────────
# Champions VGC 2026 Reg M-B  (formato: gen9championsvgc2026regmb)
# Reglas: sin Mythical ni Restricted Legendary, Level 50, Item Clause
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_TEAM_CHAMPIONS_REGMB = """
Incineroar @ Sitrus Berry
Ability: Intimidate
Level: 50
EVs: 20 HP / 11 Atk / 5 Def / 5 SpA / 20 SpD / 5 Spe
Careful Nature
- Fake Out
- Flare Blitz
- Parting Shot
- Protect

Grimmsnarl @ Light Clay
Ability: Prankster
Level: 50
EVs: 20 HP / 11 Atk / 5 Def / 5 SpA / 20 SpD / 5 Spe
Careful Nature
- Fake Out
- Spirit Break
- Light Screen
- Reflect

Garchomp @ Garchompite
Ability: Rough Skin
Level: 50
EVs: 5 HP / 32 Atk / 5 Def / 5 SpA / 5 SpD / 14 Spe
Jolly Nature
- Earthquake
- Rock Slide
- Protect
- Substitute

Annihilape @ Focus Sash
Ability: Defiant
Level: 50
EVs: 20 HP / 20 Atk / 5 Def / 5 SpA / 11 SpD / 5 Spe
Adamant Nature
- Rage Fist
- Close Combat
- Taunt
- Protect

Gholdengo @ White Herb
Ability: Good as Gold
Level: 50
EVs: 5 HP / 5 Atk / 5 Def / 32 SpA / 5 SpD / 14 Spe
Timid Nature
- Make It Rain
- Shadow Ball
- Nasty Plot
- Protect

Hatterene @ Colbur Berry
Ability: Magic Bounce
Level: 50
EVs: 20 HP / 5 Atk / 11 Def / 20 SpA / 5 SpD / 5 Spe
Quiet Nature
- Trick Room
- Dazzling Gleam
- Psychic
- Protect
"""

# ─────────────────────────────────────────────────────────────────────────────
# Equipo alternativo más defensivo (sin legendario)
# ─────────────────────────────────────────────────────────────────────────────

# Equipo alternativo más defensivo (sin legendario)
SAMPLE_TEAM_NO_RESTRICTED = """
Incineroar @ Safety Goggles
Ability: Intimidate
Level: 50
EVs: 252 HP / 4 Atk / 252 SpD
Careful Nature
- Fake Out
- Flare Blitz
- Knock Off
- Parting Shot

Amoonguss @ Rocky Helmet
Ability: Regenerator
Level: 50
EVs: 252 HP / 4 SpA / 252 SpD
Calm Nature
- Spore
- Rage Powder
- Pollen Puff
- Protect

Iron Hands @ Assault Vest
Ability: Quark Drive
Level: 50
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Drain Punch
- Heavy Slam
- Wild Charge
- Fake Out

Palafin-Hero @ Choice Band
Ability: Zero to Hero
Level: 50
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Jet Punch
- Wave Crash
- Close Combat
- Protect

Flutter Mane @ Choice Specs
Ability: Protosynthesis
Level: 50
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Moonblast
- Shadow Ball
- Mystical Fire
- Dazzling Gleam

Tornadus @ Focus Sash
Ability: Prankster
Level: 50
EVs: 252 HP / 4 SpA / 252 Spe
Timid Nature
- Tailwind
- Bleakwind Storm
- Taunt
- Protect
"""
