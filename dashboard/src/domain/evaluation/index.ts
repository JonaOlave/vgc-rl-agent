export interface BattleResult {
  battle_num: number
  result: 'WIN' | 'LOSS' | 'DRAW'
  reward: number
  our_fainted: number
  opp_fainted: number
}

export interface Evaluation {
  id: string
  timestamp: string
  model_path: string
  opponent: string
  n_battles: number
  wins: number
  losses: number
  draws: number
  win_rate: number
  mean_reward: number
  std_reward: number
  mean_our_fainted: number
  mean_opp_fainted: number
  round_number: number | null
  our_team: string | null
  opponent_archetype: string | null
  battles: BattleResult[]
}
