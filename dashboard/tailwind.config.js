/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{vue,js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: { sans: ['Inter', 'sans-serif'] },
      colors: {
        surface: '#0f172a',
        card: '#1e293b',
        border: '#334155',
        accent: '#6366f1',
        win: '#22c55e',
        loss: '#ef4444',
        draw: '#94a3b8',
      },
    },
  },
  plugins: [],
}
