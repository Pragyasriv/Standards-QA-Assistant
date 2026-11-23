/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"IBM Plex Sans"', "ui-sans-serif", "system-ui", "sans-serif"],
      },
      colors: {
        accent: {
          DEFAULT: "#1e3a5f",
          dark: "#152238",
        },
      },
      maxWidth: {
        read: "42rem",
      },
    },
  },
  plugins: [],
};