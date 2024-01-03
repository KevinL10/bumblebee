const HtmlWebpackPlugin = require("html-webpack-plugin");
const CopyPlugin = require("copy-webpack-plugin");

const path = require("path");
module.exports = {
  entry: "./bootstrap.js",
  output: {
    path: path.resolve(__dirname, "dist"),
    filename: "bootstrap.js",
  },
  mode: "production",
  plugins: [
    new CopyPlugin({
      patterns: [
        { from: "public", to: "public" },
        { from: "favicon.ico", to: "favicon.ico" },
        { from: "index.html", to: "index.html" },
      ],
    }),
  ],
  experiments: {
    asyncWebAssembly: true,
  },
 performance: {
    hints: false,
    maxEntrypointSize: 512000,
    maxAssetSize: 512000
}
};
