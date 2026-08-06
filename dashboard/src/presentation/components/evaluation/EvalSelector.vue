<template>
  <div class="bg-card rounded-xl p-5 border border-border">
    <h2 class="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">Evaluaciones</h2>

    <div v-if="loading" class="text-slate-500 text-sm">Cargando...</div>
    <ul v-else class="space-y-1.5 max-h-64 overflow-y-auto pr-1">
      <li
        v-for="ev in sortedEvaluations"
        :key="ev.id"
        @click="evalStore.selectEvaluation(ev.id)"
        class="flex items-center justify-between rounded-lg px-3 py-2 cursor-pointer transition-colors border"
        :class="selected?.id === ev.id
          ? 'bg-accent/20 border-accent/60'
          : 'bg-surface border-border hover:border-slate-500'"
      >
        <div class="flex flex-col">
          <span class="text-xs text-slate-400">{{ formatDate(ev.timestamp) }}</span>
          <span class="text-sm text-white">vs {{ ev.opponent }} · {{ ev.n_battles }} bat.</span>
        </div>
        <span
          class="text-sm font-bold"
          :class="ev.win_rate >= 0.5 ? 'text-win' : 'text-loss'"
        >
          {{ (ev.win_rate * 100).toFixed(1) }}%
        </span>
      </li>
    </ul>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { storeToRefs } from 'pinia'
import { useEvaluationStore } from '../../../application/evaluation/evaluationStore'

const evalStore = useEvaluationStore()
const { evaluations, selected, loading } = storeToRefs(evalStore)

const sortedEvaluations = computed(() =>
  [...evaluations.value].sort(
    (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime(),
  ),
)

function formatDate(iso: string) {
  return new Date(iso).toLocaleString('es-CL', {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  })
}
</script>
