<template>
  <div class="bg-card rounded-xl p-5 border border-border">
    <h2 class="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">Checkpoints</h2>

    <div v-if="loading" class="text-slate-500 text-sm">Cargando...</div>
    <div v-else-if="checkpoints.length === 0" class="text-slate-500 text-sm">Sin checkpoints</div>

    <ul v-else class="space-y-2 max-h-96 overflow-y-auto pr-1">
      <li
        v-for="cp in checkpoints"
        :key="cp.name"
        class="flex items-center justify-between rounded-lg px-3 py-2 bg-surface border border-border"
      >
        <div class="flex flex-col">
          <span class="text-sm font-medium text-white">
            {{ cp.name === 'final' ? '★ final' : `${(cp.steps / 1000).toFixed(0)}K steps` }}
          </span>
          <span class="text-xs text-slate-400">{{ formatDate(cp.timestamp) }}</span>
        </div>
        <span class="text-xs text-slate-500">{{ cp.size_mb }} MB</span>
      </li>
    </ul>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'
import { storeToRefs } from 'pinia'
import { useTrainingStore } from '../../../application/training/trainingStore'

const store = useTrainingStore()
const { checkpoints, loading } = storeToRefs(store)

onMounted(() => store.fetchCheckpoints())

function formatDate(iso: string) {
  return new Date(iso).toLocaleString('es-CL', {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  })
}
</script>
