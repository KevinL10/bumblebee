import * as wasm from "webvit";

let imageBytes = null;
let model = null;
let chart = null;

const images = ["airplane1", "cat1"]

async function fetchRandomImage() {
  const img = images[Math.floor(Math.random() * images.length)]
  const res = await fetch(`/public/${img}.png`, { cache: "force-cache" });
  const bytes = new Uint8Array(await res.arrayBuffer());
  imageBytes = bytes;

  // set image in html
  var base64String = btoa(
    String.fromCharCode.apply(null, new Uint8Array(imageBytes))
  );
  document.getElementById("image").src =
    "data:image/png;base64," + base64String;

//   return ;
}

async function init() {
  const res = await fetch("/public/weights.safetensors", {
    cache: "force-cache",
  });
  const weights = new Uint8Array(await res.arrayBuffer());

  model = new wasm.WasmModel(weights);
}

async function predict() {
  if (imageBytes === null) {
    alert("cannot predict without image");
    return;
  }

  // let image = await fetchImage();
  console.log("fetched image");
  console.log(model.info());

  console.log(imageBytes);
  console.log(imageBytes.length);

  console.log("predicting");
  let probs = model.predict_image(imageBytes);
  console.log(probs);
  setupChart(probs);
}

function setupChart(probabilities) {
  if (chart !== null) {
    chart.destroy();
  }

  const ctx = document.getElementById("chart");

  chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
      ],
      datasets: [
        {
          label: "Probability of class",
          data: probabilities,
          borderWidth: 1,
        },
      ],
    },
    options: {
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
        },
      },
    },
  });
}

let randomButton = document.getElementById("random");
let predictButton = document.getElementById("predict");

// TODO: fix orders of init
init();

randomButton.addEventListener("click", () => {
  fetchRandomImage();
});

predictButton.addEventListener("click", () => {
  predict();
});
