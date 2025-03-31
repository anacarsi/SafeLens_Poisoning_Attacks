// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const path = require('path');
const CopyPlugin = require("copy-webpack-plugin");
const CopyWebpackPlugin = require('copy-webpack-plugin');

module.exports = () => {
    return {
        target: ['web'],
        entry: path.resolve(__dirname, 'main.js'),
        output: {
            path: path.resolve(__dirname, 'dist'),
            filename: 'bundle.min.js',
            library: {
                type: 'umd'
            }
        },
        plugins: [
            new CopyPlugin({
                // Use copy plugin to copy *.wasm to output folder.
                patterns: [{ from: 'node_modules/onnxruntime-web/dist/*.wasm', to: '[name][ext]' }]
            }),
            new CopyWebpackPlugin({
                patterns: [
                    {
                        from: 'src/model',  // Source directory
                        to: 'model'         // Output directory
                    }
                ]
            })
        ],
        mode: 'production'
    }
};