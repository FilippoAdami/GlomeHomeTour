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
          navy: '#0A1628',
          'navy-mid': '#1A2B48',
          blue: '#1E56A0',
          'blue-light': '#3B82F6',
          gold: '#F5A623',
          'off-white': '#F8F9FC',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
