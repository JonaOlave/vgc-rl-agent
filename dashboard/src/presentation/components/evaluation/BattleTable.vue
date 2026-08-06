<template>
  <div class="bg-card rounded-xl p-5 border border-border">
    <div class="flex items-center justify-between mb-4">
      <h2 class="text-sm font-semibold text-slate-400 uppercase tracking-wider">Batallas</h2>
      <div v-if="selected" class="flex gap-3 text-xs">
        <span class="text-slate-400">
          Oponente: <span class="text-white font-medium">{{ selected.opponent }}</span>
        </span>
        <span class="text-win font-semibold">{{ selected.wins }}V</span>
        <span class="text-loss font-semibold">{{ selected.losses }}D</span>
        <span class="text-slate-400 font-semibold">
          {{ (selected.win_rate * 100).toFixed(1) }}%
        </span>
      </div>
    </div>

    <div v-if="!selected" class="text-slate-500 text-sm">Selecciona una evaluación</div>

    <div v-else-if="selected.battles.length === 0" class="text-slate-500 text-sm">
      Sin detalle de batallas (evaluación histórica)
    </div>

    <div v-else class="overflow-auto max-h-80">
      <table class="w-full text-sm">
        <thead>
          <tr class="text-xs text-slate-500 border-b border-border">
            <th class="text-left py-2 pr-4">#</th>
            <th class="text-left py-2 pr-4">Resultado</th>
            <th class="text-right py-2 pr-4">Reward</th>
            <th class="text-right py-2 pr-4">Caídos nuestros</th>
            <th class="text-right py-2">Caídos oponente</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="b in selected.battles"
            :key="b.battle_num"
            class="border-b border-border/50 hover:bg-surface/50 transition-colors"
          >
            <td class="py-1.5 pr-4 text-slate-400">{{ b.battle_num }}</td>
            <td class="py-1.5 pr-4">
              <span
                class="px-2 py-0.5 rounded text-xs font-semibold"
                :class="{
                  'bg-win/20 text-win': b.result === 'WIN',
                  'bg-loss/20 text-loss': b.result === 'LOSS',
                  'bg-slate-600/20 text-slate-400': b.result === 'DRAW',
                }"
              >{{ b.result }}</span>
            </td>
            <td class="py-1.5 pr-4 text-right font-mono"
              :class="b.reward >= 0 ? 'text-win' : 'text-loss'">
              {{ b.reward >= 0 ? '+' : '' }}{{ b.reward.toFixed(2) }}
            </td>
            <td class="py-1.5 pr-4 text-right text-slate-300">{{ b.our_fainted }}/6</td>
            <td class="py-1.5 text-right text-slate-300">{{ b.opp_fainted }}/6</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup lang="ts">
import { storeToRefs } from 'pinia'
import { useEvaluationStore } from '../../../application/evaluation/evaluationStore'

const store = useEvaluationStore()
const { selected } = storeToRefs(store)
</script>
