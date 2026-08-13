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
# NOTA: este equipo usa EVs/items de VGC estándar y NO es legal en
# gen9championsvgc2026regmb — ese formato corre el mod "champions" de PS
# (dex y sistema de EVs propios: máx. 32 por stat, 66 en total, y varios
# items modernos como Assault Vest / Rocky Helmet / Choice Band están
# baneados). Ver SAMPLE_TEAM_TRICKROOM / SAMPLE_TEAM_TAILWIND para
# equipos sí válidos en ese formato.

# ─────────────────────────────────────────────────────────────────────────────
# Pool de oponentes alternativos para Champions VGC 2026 Reg M-B
# Validados con: ./pokemon-showdown validate-team gen9championsvgc2026regmb
# Pensados para variar tipos/resistencias/estrategia frente al equipo propio
# (SAMPLE_TEAM_CHAMPIONS_REGMB) y evitar que el entrenamiento sea siempre
# un mirror match.
# ─────────────────────────────────────────────────────────────────────────────

# Núcleo de Trick Room (Slowking/Farigiraf/Runerigus como setters + pegadores)
SAMPLE_TEAM_TRICKROOM = """
Slowking @ Leftovers
Ability: Regenerator
Level: 50
EVs: 20 HP / 16 Def / 20 SpD / 10 Spe
Calm Nature
- Trick Room
- Scald
- Psychic
- Helping Hand

Farigiraf @ Sitrus Berry
Ability: Armor Tail
Level: 50
EVs: 20 HP / 26 SpA / 20 SpD
Quiet Nature
- Hyper Voice
- Protect
- Psychic
- Trick Room

Kangaskhan @ Life Orb
Ability: Scrappy
Level: 50
EVs: 20 HP / 26 Atk / 20 Spe
Adamant Nature
- Double-Edge
- Drain Punch
- Fake Out
- Protect

Runerigus @ Colbur Berry
Ability: Wandering Spirit
Level: 50
EVs: 20 HP / 26 Def / 20 SpD
Impish Nature
- Body Press
- Poltergeist
- Trick Room
- Will-O-Wisp

Sableye @ Mental Herb
Ability: Prankster
Level: 50
EVs: 26 HP / 20 SpD / 20 Spe
Careful Nature
- Encore
- Fake Out
- Knock Off
- Thunder Wave

Toxicroak @ Focus Sash
Ability: Dry Skin
Level: 50
EVs: 16 HP / 26 Atk / 24 Spe
Jolly Nature
- Close Combat
- Fake Out
- Gunk Shot
- Protect
"""

# Ofensiva de Tailwind (Pelipper/Whimsicott/Dragonite/Corviknight/Weavile/Swampert)
SAMPLE_TEAM_TAILWIND = """
Pelipper @ Wide Lens
Ability: Drizzle
Level: 50
EVs: 26 HP / 20 SpA / 20 Spe
Timid Nature
- Tailwind
- Hydro Pump
- Protect
- Wide Guard

Dragonite @ Lum Berry
Ability: Inner Focus
Level: 50
EVs: 10 HP / 20 Atk / 16 SpA / 20 Spe
Naive Nature
- Dragon Claw
- Fire Blast
- Protect
- Tailwind

Whimsicott @ Focus Sash
Ability: Prankster
Level: 50
EVs: 20 HP / 20 SpA / 26 Spe
Timid Nature
- Encore
- Moonblast
- Protect
- Tailwind

Corviknight @ Leftovers
Ability: Mirror Armor
Level: 50
EVs: 26 HP / 26 Def / 14 Spe
Impish Nature
- Body Press
- Brave Bird
- Roost
- Tailwind

Weavile @ Life Orb
Ability: Pickpocket
Level: 50
EVs: 10 HP / 26 Atk / 30 Spe
Jolly Nature
- Fake Out
- Knock Off
- Protect
- Triple Axel

Swampert @ Sitrus Berry
Ability: Torrent
Level: 50
EVs: 26 HP / 20 Def / 20 SpD
Careful Nature
- Wave Crash
- High Horsepower
- Icy Wind
- Knock Off
"""

# Equipo de lluvia — diseñado por el usuario (criterio competitivo real),
# validado con: ./pokemon-showdown validate-team gen9championsvgc2026regmb
SAMPLE_TEAM_RAIN = """
Archaludon @ Leftovers
Ability: Stamina
Level: 50
EVs: 32 HP / 11 SpA / 23 SpD
Calm Nature
- Protect
- Dragon Pulse
- Flash Cannon
- Electro Shot

Swampert @ Swampertite
Ability: Damp
Level: 50
EVs: 18 HP / 32 Atk / 16 Spe
Adamant Nature
- Protect
- High Horsepower
- Ice Punch
- Wave Crash

Blaziken @ Life Orb
Ability: Speed Boost
Level: 50
EVs: 11 HP / 17 Atk / 1 Def / 15 SpA / 22 Spe
Naive Nature
- Detect
- Brave Bird
- Heat Wave
- Close Combat

Pelipper @ Damp Rock
Ability: Drizzle
Level: 50
EVs: 31 HP / 2 SpA / 32 SpD / 1 Spe
Modest Nature
- Tailwind
- Wide Guard
- Weather Ball
- Hurricane

Floette-Eternal (F) @ Floettite
Ability: Flower Veil
Level: 50
EVs: 31 HP / 14 Def / 21 Spe
Timid Nature
- Protect
- Dazzling Gleam
- Moonblast
- Calm Mind

Liepard @ Focus Sash
Ability: Prankster
Level: 50
EVs: 32 HP / 2 Def / 32 Spe
Jolly Nature
- Fake Out
- Copycat
- Encore
- Foul Play
"""

# Equipo de arena — diseñado por el usuario (criterio competitivo real).
# Revisión 2026-08-11: Tyranitar pasó de Tyranitarite a Chople Berry — Sand
# Stream ya es su habilidad base (no depende de mega, a diferencia del Sol),
# así que la piedra mega era un slot de ítem desperdiciado contra un bot que
# nunca mega-evoluciona.
# Validado con: ./pokemon-showdown validate-team gen9championsvgc2026regmb
SAMPLE_TEAM_SAND = """
Tyranitar @ Chople Berry
Ability: Sand Stream
Level: 50
EVs: 32 HP / 15 Atk / 1 Def / 1 SpD / 17 Spe
Adamant Nature
- Rock Slide
- Knock Off
- Ice Punch
- Protect

Sinistcha @ Sitrus Berry
Ability: Hospitality
Level: 50
EVs: 30 HP / 24 Def / 1 SpA / 11 SpD
Relaxed Nature
- Protect
- Rage Powder
- Matcha Gotcha
- Trick Room

Corviknight @ Leftovers
Ability: Mirror Armor
Level: 50
EVs: 27 HP / 11 Atk / 1 Def / 5 SpD / 22 Spe
Careful Nature
- Bulk Up
- Brave Bird
- Roost
- Tailwind

Rotom-Wash @ Choice Scarf
Ability: Levitate
Level: 50
EVs: 2 Def / 32 SpA / 32 Spe
Modest Nature
- Hydro Pump
- Thunderbolt
- Volt Switch
- Electroweb

Excadrill @ Focus Sash
Ability: Sand Rush
Level: 50
EVs: 2 HP / 32 Atk / 32 Spe
Jolly Nature
- Rock Slide
- Iron Head
- Earthquake
- Protect

Typhlosion-Hisui @ Life Orb
Ability: Frisk
Level: 50
EVs: 1 HP / 32 SpA / 1 SpD / 32 Spe
Timid Nature
- Infernal Parade
- Overheat
- Eruption
- Protect
"""

# Equipo de sol — diseñado por el usuario (criterio competitivo real).
# Revisión 2026-08-11: se reemplazó a Charizard (Mega Charizard Y por Drought)
# por Ninetales @ Focus Sash con Drought nativa, porque SimpleHeuristicsPlayer
# (el bot oponente) nunca mega-evoluciona (ver poke_env/player/baselines.py) —
# el plan original de Drought vía mega nunca se activaba cuando este equipo lo
# pilotaba el bot. Annihilape pasa de Focus Sash a Expert Belt.
# Validado con: ./pokemon-showdown validate-team gen9championsvgc2026regmb
# (nota: este mod no permite personalizar IVs — quedan fijas en 31, así que
# no se incluye la línea "IVs:" del equipo original).
SAMPLE_TEAM_SUN = """
Garchomp @ Choice Scarf
Ability: Rough Skin
Level: 50
EVs: 2 HP / 32 Atk / 32 Spe
Adamant Nature
- Earthquake
- Dragon Claw
- Stomping Tantrum
- Rock Slide

Venusaur @ Life Orb
Ability: Chlorophyll
Level: 50
EVs: 2 HP / 32 SpA / 32 Spe
Modest Nature
- Leaf Storm
- Sludge Bomb
- Earth Power
- Protect

Incineroar @ Sitrus Berry
Ability: Intimidate
Level: 50
EVs: 32 HP / 2 Atk / 32 Def
Impish Nature
- Flare Blitz
- Parting Shot
- Protect
- Fake Out

Ninetales @ Focus Sash
Ability: Drought
Level: 50
EVs: 2 HP / 32 SpA / 32 Spe
Timid Nature
- Heat Wave
- Solar Beam
- Overheat
- Protect

Toxapex @ Leftovers
Ability: Regenerator
Level: 50
EVs: 32 HP / 5 Def / 29 SpD
Relaxed Nature
- Infestation
- Toxic
- Wide Guard
- Baneful Bunker

Annihilape @ Expert Belt
Ability: Defiant
Level: 50
EVs: 32 HP / 2 Atk / 32 Spe
Jolly Nature
- Close Combat
- Phantom Force
- Rock Tomb
- Protect
"""

# Pool por defecto para variar al oponente durante el entrenamiento —
# incluye el propio equipo (mirror match ocasional) + los diseñados aparte.
OPPONENT_TEAM_POOL_CHAMPIONS_REGMB = [
    SAMPLE_TEAM_CHAMPIONS_REGMB,
    SAMPLE_TEAM_TRICKROOM,
    SAMPLE_TEAM_TAILWIND,
    SAMPLE_TEAM_RAIN,
    SAMPLE_TEAM_SAND,
    SAMPLE_TEAM_SUN,
]
