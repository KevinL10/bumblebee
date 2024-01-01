import * as wasm from "webvit";

// wasm.greet("abcd");



// fetch()j
// const weights = Uint8Array(await res.arrayBuffer());
async function fetchImage() {

    const res = await fetch('/public/cat1.png', {cache: "force-cache"});
    const image = new Uint8Array(await res.arrayBuffer());

    return image;
}

async function fetchWeights() {

    const res = await fetch('/public/weights.safetensors', {cache: "force-cache"});
    const weights = new Uint8Array(await res.arrayBuffer());
    return weights
}

//
async function main() {

    const weights = await fetchWeights();

    console.log('fetched weights')
    const model = new wasm.WasmModel(weights);

    let image = await fetchImage();
    console.log('fetched image')
    console.log(model.info())

    console.log(image.length);

    console.log('predicting')
    let probs = model.predict_image(image);
    console.log(probs);
}


let button = document.getElementById("button");

button.addEventListener("click", () => {
    main();
});
