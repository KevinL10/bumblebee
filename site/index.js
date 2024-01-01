import * as wasm from "webvit";

wasm.greet("abcd");


const model = new wasm.WasmModel();


console.log(model.name())
