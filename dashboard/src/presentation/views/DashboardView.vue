<template>
  <div class="min-h-screen bg-surface text-white">
    <!-- Header -->
    <header class="border-b border-border px-6 py-4 flex items-center gap-3">
      <span class="text-2xl">⚔️</span>
      <div>
        <h1 class="text-lg font-bold text-white">VGC RL Agent</h1>
        <p class="text-xs text-slate-400">Training Dashboard</p>
      </div>
    </header>

    <main class="p-6 space-y-6">
      <!-- Stats cards row -->
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Mejor win rate" :value="`${bestWinRate}%`" color="text-win" />
        <StatCard label="vs Random" :value="`${latestRandom}%`" color="text-accent" />
        <StatCard label="vs Heurístico" :value="`${latestHeuristic}%`" color="text-orange-400" />
        <StatCard label="Evaluaciones" :value="String(evaluations.length)" color="text-slate-300" />
      </div>

      <!-- Chart + sidebar -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div class="lg:col-span-2">
          <WinRateChart />
        </div>
        <div class="flex flex-col gap-4">
          <CheckpointTimeline />
          <EvalLauncher />
        </div>
      </div>

      <!-- Evaluation detail -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div class="lg:col-span-1">
          <EvalSelector />
        </div>
        <div class="lg:col-span-2">
          <BattleTable />
        </div>
      </div>
    </main>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { storeToRefs } from 'pinia'
import { useEvaluationStore } from '../../application/evaluation/evaluationStore'
import { useTrainingStore } from '../../application/training/trainingStore'
import WinRateChart from '../components/evaluation/WinRateChart.vue'
import BattleTable from '../components/evaluation/BattleTable.vue'
import EvalLauncher from '../components/evaluation/EvalLauncher.vue'
import EvalSelector from '../components/evaluation/EvalSelector.vue'
import CheckpointTimeline from '../components/training/CheckpointTimeline.vue'
import StatCard from '../components/StatCard.vue'

const evalStore = useEvaluationStore()
const trainingStore = useTrainingStore()
const { evaluations } = storeToRefs(evalStore)

onMounted(() => {
  evalStore.fetchEvaluations()
  trainingStore.fetchCheckpoints()
})

const bestWinRate = computed(() => {
  if (!evaluations.value.length) return '—'
  return (Math.max(...evaluations.value.map(e => e.win_rate)) * 100).toFixed(1)
})

const latestRandom = computed(() => {
  const random = evaluations.value.filter(e => e.opponent === 'random')
  if (!random.length) return '—'
  const latest = random.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())[0]
  return (latest.win_rate * 100).toFixed(1)
})

const latestHeuristic = computed(() => {
  const heuristic = evaluations.value.filter(e => e.opponent === 'heuristic')
  if (!heuristic.length) return '—'
  const latest = heuristic.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())[0]
  return (latest.win_rate * 100).toFixed(1)
})
</script>
