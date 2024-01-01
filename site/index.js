import * as wasm from "webvit";

// wasm.greet("abcd");



// fetch()j
// const weights = Uint8Array(await res.arrayBuffer());
const weights = fetch('/public/weights.safetensors', {cache: "force-cache"})
    .then(res => res.arrayBuffer())
    .then(arrayBuffer => new Uint8Array(arrayBuffer))
    .then(weights => {
        return weights;
    });

weights.then(weights => {
    console.log(weights.length);
    const model = new wasm.WasmModel(weights);
})



// console.log(model.info())
