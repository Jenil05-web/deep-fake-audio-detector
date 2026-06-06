/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class', // Default is dark, but supporting toggling if needed
  theme: {
    extend: {
      colors: {
        cyber: {
          bg: "#040406",
          card: "#0b0c10",
          border: "#181b24",
          borderGlow: "#00ff8822",
          text: "#e2e8f0",
          muted: "#64748b",
          neonGreen: "#00ff88",
          neonRed: "#ff3366",
          neonBlue: "#00e5ff",
          panel: "#0e111a",
        }
      },
      fontFamily: {
        mono: ['"Space Mono"', 'monospace'],
        sans: ['"Inter"', 'sans-serif'],
        syne: ['"Syne"', 'sans-serif'],
      },
      boxShadow: {
        'glow-green': '0 0 15px rgba(0, 255, 136, 0.4)',
        'glow-red': '0 0 15px rgba(255, 51, 102, 0.4)',
        'glow-blue': '0 0 15px rgba(0, 229, 255, 0.4)',
        'glass-green': 'inset 0 1px 1px rgba(255, 255, 255, 0.05), 0 0 20px rgba(0, 255, 136, 0.1)',
        'glass-red': 'inset 0 1px 1px rgba(255, 255, 255, 0.05), 0 0 20px rgba(255, 51, 102, 0.1)',
      },
      animation: {
        'scanline': 'scanline 6s linear infinite',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'matrix-fade': 'matrixFade 20s linear infinite',
      },
      keyframes: {
        scanline: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100%)' },
        },
        matrixFade: {
          '0%': { opacity: '0.02' },
          '50%': { opacity: '0.06' },
          '100%': { opacity: '0.02' },
        }
      }
    },
  },
  plugins: [],
}
