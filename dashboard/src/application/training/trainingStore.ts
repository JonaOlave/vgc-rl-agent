import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { Checkpoint } from '../../domain/training'
import { trainingApi } from '../../infrastructure/api/trainingApi'

export const useTrainingStore = defineStore('training', () => {
  const checkpoints = ref<Checkpoint[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function fetchCheckpoints() {
    loading.value = true
    error.value = null
    try {
      checkpoints.value = await trainingApi.getCheckpoints()
    } catch (e: any) {
      error.value = e.message
    } finally {
      loading.value = false
    }
  }

  return { checkpoints, loading, error, fetchCheckpoints }
})
