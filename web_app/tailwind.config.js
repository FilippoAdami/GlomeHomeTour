/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Glome brand palette — clean whites + deep navy accent
        glome: {
          // Canvas & Neutral tones (Warm Cream / Alabaster)
          cream: '#FBF9F5',
          'cream-surface': '#FFFFFF',
          'cream-dark': '#F2EEE7',
          stone: '#E8E2D8',
          charcoal: '#14171A',
          'charcoal-light': '#2A2F35',
          muted: '#68707C',

          // Product 1: Glome Home Tour (Warm Earthy Orange / Terracotta)
          orange: '#D95D28',
          'orange-light': '#E87342',
          'orange-dark': '#B84717',
          'orange-tint': '#FDF2EC',

          // Product 2: Glome Home Design (Moss Green)
          moss: '#446B4B',
          'moss-light': '#56845E',
          'moss-dark': '#325037',
          'moss-tint': '#F0F5F1',

          // Product 3: Glome Build (Blueprint Blue)
          blueprint: '#1E5494',
          'blueprint-light': '#2D6CB4',
          'blueprint-dark': '#163D6C',
          'blueprint-tint': '#EEF4FB',

          // Legacy mappings for backwards-compatibility
          navy: '#14171A',
          'navy-mid': '#2A2F35',
          blue: '#D95D28', // maps to primary action (orange)
          'blue-light': '#E87342',
          gold: '#D95D28',
          'off-white': '#FBF9F5',
        }
      },
      boxShadow: {
        'glass-sm': '0 4px 20px -2px rgba(20, 23, 26, 0.05), inset 0 1px 1px 0 rgba(255, 255, 255, 0.7)',
        'glass-md': '0 12px 36px -4px rgba(20, 23, 26, 0.08), inset 0 1px 1px 0 rgba(255, 255, 255, 0.85)',
        'glass-lg': '0 24px 64px -8px rgba(20, 23, 26, 0.12), inset 0 1px 2px 0 rgba(255, 255, 255, 0.9)',
        'glass-glow-orange': '0 0 40px -10px rgba(217, 93, 40, 0.25), 0 16px 36px -4px rgba(20, 23, 26, 0.08)',
        'glass-glow-moss': '0 0 40px -10px rgba(68, 107, 75, 0.25), 0 16px 36px -4px rgba(20, 23, 26, 0.08)',
        'glass-glow-blueprint': '0 0 40px -10px rgba(30, 84, 148, 0.25), 0 16px 36px -4px rgba(20, 23, 26, 0.08)',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
