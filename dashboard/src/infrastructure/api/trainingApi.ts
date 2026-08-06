import { apiClient } from './client'
import type { Checkpoint } from '../../domain/training'

export const trainingApi = {
  getCheckpoints(): Promise<Checkpoint[]> {
    return apiClient.get<Checkpoint[]>('/training/checkpoints').then(r => r.data)
  },
}
