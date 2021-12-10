const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const WasmPackPlugin = require('@wasm-tool/wasm-pack-plugin');
const port = process.env.PORT || 3000

module.exports = {
    entry: path.join(process.cwd(), "www", "index.js"),
    output: {
        path: path.join(process.cwd(), "build"),
        filename: "index.bundle.js"
    },
    mode: process.env.NODE_ENV || "development",
    resolve: {
        modules: [path.resolve(process.cwd(), "www"), "node_modules"]
    },
    devServer: {
        static: path.join(process.cwd(), "public"),
        host: 'localhost',
        port: port,
        historyApiFallback: true,
        open: true,
        webSocketServer: 'ws',
    },
    module: {
        rules: [
            { 
                test: /\.(js|jsx)$/, 
                exclude: /node_modules/, 
                use: ["babel-loader"] 
            },
            {
                test: /\.(css|scss)$/,
                use: ["style-loader", "css-loader"],
            },
            { 
                test: /\.(jpg|jpeg|png|gif|mp3|svg)$/,
                use: ["file-loader"] 
            },
        ],
    },
    plugins: [
        new HtmlWebpackPlugin({
            template: path.join(process.cwd(), "public", "index.html"),
        }),
        new WasmPackPlugin({
            crateDirectory: process.cwd(),
            outName: 'infer',
            outDir: 'public',
        }),
    ],
    devtool: 'inline-source-map',
};