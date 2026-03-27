/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        warm: {
          50:  '#FAFAF7',
          100: '#F5F2ED',
          150: '#F0EDE6',
          200: '#EBE7E0',
          300: '#E8E4DD',
          400: '#D4CFC6',
          500: '#A39E95',
          600: '#78736A',
          700: '#5C5750',
          800: '#3D3935',
          900: '#2C2925',
        },
        accent: {
          DEFAULT: '#8B7355',
          light: '#C4A882',
          dark: '#6B5A42',
          subtle: 'rgba(139, 115, 85, 0.08)',
        },
        success: '#6B9B6B',
      },
      fontFamily: {
        display: ['Fraunces', 'Georgia', 'serif'],
        sans: ['Outfit', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      animation: {
        'fade-in': 'fadeIn 0.3s ease-out both',
        'fade-in-up': 'fadeInUp 0.4s ease-out both',
      },
      keyframes: {
        fadeIn: {
          from: { opacity: '0' },
          to:   { opacity: '1' },
        },
        fadeInUp: {
          from: { opacity: '0', transform: 'translateY(8px)' },
          to:   { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
}
