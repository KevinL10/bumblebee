import * as wasm from "webvit";


let imageBytes = null;
let model = null;

async function fetchImage() {
    const res = await fetch('/public/airplane1.png', {cache: "force-cache"});
    const image = new Uint8Array(await res.arrayBuffer());
    imageBytes = image;

    // set image in html
    var base64String = btoa(String.fromCharCode.apply(null, new Uint8Array(image)));
    document.getElementById("image").src = "data:image/png;base64," + base64String;

    return image;
}

async function init() {
    const res = await fetch('/public/weights.safetensors', {cache: "force-cache"});
    const weights = new Uint8Array(await res.arrayBuffer());
    
    model = new wasm.WasmModel(weights);
}

async function predict() {
    if(imageBytes === null) {
        alert("cannot predict without image")
        return;
    }

    // let image = await fetchImage();
    console.log('fetched image')
    console.log(model.info())

    console.log(imageBytes);
    console.log(imageBytes.length);

    console.log('predicting')
    let probs = model.predict_image(imageBytes);
    console.log(probs);
}

let randomButton = document.getElementById("random");
let predictButton = document.getElementById("predict");

// TODO: fix orders of init
init()

randomButton.addEventListener("click", () => {
    fetchImage();
});

predictButton.addEventListener("click", () => {
    predict();
});
