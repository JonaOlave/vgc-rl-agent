<template>
  <div class="bg-card rounded-xl p-5 border border-border">
    <h2 class="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">Nueva evaluación</h2>

    <div class="space-y-3">
      <div>
        <label class="block text-xs text-slate-400 mb-1">Modelo</label>
        <select v-model="form.model_path" class="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-accent">
          <option value="models/vgc/final">★ final</option>
          <option v-for="cp in checkpointOptions" :key="cp.name" :value="cp.path">
            {{ (cp.steps / 1000).toFixed(0) }}K steps
          </option>
        </select>
      </div>

      <div>
        <label class="block text-xs text-slate-400 mb-1">Oponente</label>
        <select v-model="form.opponent" class="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-accent">
          <option value="random">Random</option>
          <option value="heuristic">Heurístico</option>
        </select>
      </div>

      <div>
        <label class="block text-xs text-slate-400 mb-1">Batallas</label>
        <input
          v-model.number="form.n_battles"
          type="number"
          min="5"
          max="100"
          class="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-accent"
        />
      </div>

      <button
        @click="launch"
        :disabled="running"
        class="w-full py-2.5 rounded-lg font-semibold text-sm transition-all"
        :class="running
          ? 'bg-accent/40 text-slate-400 cursor-not-allowed'
          : 'bg-accent hover:bg-accent/80 text-white'"
      >
        <span v-if="running" class="flex items-center justify-center gap-2">
          <svg class="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
          </svg>
          Evaluando...
        </span>
        <span v-else>Iniciar evaluación</span>
      </button>

      <p v-if="error" class="text-loss text-xs mt-1">{{ error }}</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { reactive, computed } from 'vue'
import { storeToRefs } from 'pinia'
import { useEvaluationStore } from '../../../application/evaluation/evaluationStore'
import { useTrainingStore } from '../../../application/training/trainingStore'

const evalStore = useEvaluationStore()
const trainingStore = useTrainingStore()
const { running, error } = storeToRefs(evalStore)
const { checkpoints } = storeToRefs(trainingStore)

const checkpointOptions = computed(() =>
  checkpoints.value.filter(c => c.name !== 'final').sort((a, b) => b.steps - a.steps),
)

const form = reactive({
  model_path: 'models/vgc/final',
  opponent: 'random',
  n_battles: 30,
})

function launch() {
  evalStore.runEvaluation({
    model_path: form.model_path,
    opponent: form.opponent,
    n_battles: form.n_battles,
    use_champions_team: true,
  })
}
</script>
