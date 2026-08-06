import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { Evaluation } from '../../domain/evaluation'
import { evaluationApi, type RunEvaluationParams } from '../../infrastructure/api/evaluationApi'

export const useEvaluationStore = defineStore('evaluation', () => {
  const evaluations = ref<Evaluation[]>([])
  const selected = ref<Evaluation | null>(null)
  const running = ref(false)
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function fetchEvaluations() {
    loading.value = true
    error.value = null
    try {
      evaluations.value = await evaluationApi.getEvaluations()
      if (evaluations.value.length > 0 && !selected.value) {
        selected.value = evaluations.value[evaluations.value.length - 1]
      }
    } catch (e: any) {
      error.value = e.message
    } finally {
      loading.value = false
    }
  }

  async function runEvaluation(params: RunEvaluationParams) {
    running.value = true
    error.value = null
    try {
      const result = await evaluationApi.runEvaluation(params)
      evaluations.value.push(result)
      selected.value = result
    } catch (e: any) {
      error.value = e.message
    } finally {
      running.value = false
    }
  }

  function selectEvaluation(id: string) {
    selected.value = evaluations.value.find(e => e.id === id) ?? null
  }

  return { evaluations, selected, running, loading, error, fetchEvaluations, runEvaluation, selectEvaluation }
})
