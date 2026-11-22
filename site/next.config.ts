import type { NextConfig } from "next";

// a fully static site: `next build` writes plain html/css/js into out/, and the model runs in the visitor's browser
const nextConfig: NextConfig = {
  output: "export",
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
