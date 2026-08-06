import { apiClient } from './client'
import type { Evaluation } from '../../domain/evaluation'

export interface RunEvaluationParams {
  model_path: string
  opponent: string
  n_battles: number
  use_champions_team: boolean
}

export const evaluationApi = {
  getEvaluations(): Promise<Evaluation[]> {
    return apiClient.get<Evaluation[]>('/evaluation/results').then(r => r.data)
  },
  getEvaluation(id: string): Promise<Evaluation> {
    return apiClient.get<Evaluation>(`/evaluation/results/${id}`).then(r => r.data)
  },
  runEvaluation(params: RunEvaluationParams): Promise<Evaluation> {
    return apiClient.post<Evaluation>('/evaluation/run', params).then(r => r.data)
  },
}
