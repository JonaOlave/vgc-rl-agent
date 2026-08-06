<template>
  <div class="bg-card rounded-xl p-4 border border-border">
    <p class="text-xs text-slate-400 mb-1">{{ label }}</p>
    <p class="text-2xl font-bold" :class="color">{{ value }}</p>
  </div>
</template>

<script setup lang="ts">
defineProps<{ label: string; value: string; color?: string }>()
</script>
