<template>
  <div class="bg-card rounded-xl p-5 border border-border">
    <h2 class="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">Win Rate histórico</h2>
    <Line :data="chartData" :options="chartOptions" />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { Line } from 'vue-chartjs'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js'
import { storeToRefs } from 'pinia'
import { useEvaluationStore } from '../../../application/evaluation/evaluationStore'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend)

const store = useEvaluationStore()
const { evaluations } = storeToRefs(store)

const randomColor = '#6366f1'
const heuristicColor = '#f97316'

const chartData = computed(() => {
  const sorted = [...evaluations.value].sort(
    (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime(),
  )

  const randomEvals = sorted.filter(e => e.opponent === 'random')
  const heuristicEvals = sorted.filter(e => e.opponent === 'heuristic')

  const allLabels = sorted.map(e =>
    new Date(e.timestamp).toLocaleString('es-CL', {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    }),
  )

  const toDataset = (evals: typeof sorted, color: string, label: string) => ({
    label,
    data: sorted.map(s => {
      const match = evals.find(e => e.id === s.id)
      return match ? Math.round(match.win_rate * 100) : null
    }),
    borderColor: color,
    backgroundColor: color + '33',
    pointBackgroundColor: color,
    pointRadius: 5,
    pointHoverRadius: 7,
    tension: 0.3,
    spanGaps: true,
  })

  return {
    labels: allLabels,
    datasets: [
      toDataset(randomEvals, randomColor, 'vs Random'),
      toDataset(heuristicEvals, heuristicColor, 'vs Heurístico'),
    ],
  }
})

const chartOptions = {
  responsive: true,
  interaction: { mode: 'index' as const, intersect: false },
  plugins: {
    legend: { labels: { color: '#94a3b8', font: { size: 12 } } },
    tooltip: {
      callbacks: {
        label: (ctx: any) => ctx.raw !== null ? `${ctx.dataset.label}: ${ctx.raw}%` : '',
      },
    },
  },
  scales: {
    x: { ticks: { color: '#64748b', maxRotation: 30 }, grid: { color: '#1e293b' } },
    y: {
      min: 0, max: 100,
      ticks: { color: '#64748b', callback: (v: any) => `${v}%` },
      grid: { color: '#334155' },
    },
  },
}
</script>
