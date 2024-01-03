import * as wasm from "bumblebee";

let imageBytes = null;
let model = null;
let chart = null;

const images = [
  "bumblebee.png",
  "airplane2.png",
  "truck.jpg",
  "bird1.webp",
  "frog2.jpg",
  "dog1.jpeg",
  "airplane1.png",
  "cat.png",
  "automobile2.png",
  "airplane3.png",
  "bird2.webp",
  "frog1.png",
  "automobile1.png",
];

async function newImagePredict() {
  const img = images[Math.floor(Math.random() * images.length)];
  const res = await fetch(`/public/${img}`, { cache: "force-cache" });
  const bytes = new Uint8Array(await res.arrayBuffer());
  imageBytes = bytes;

  // set image in html
  var base64String = btoa(
    String.fromCharCode.apply(null, new Uint8Array(imageBytes))
  );
  document.getElementById("image").src =
    "data:image/png;base64," + base64String;

  predict();
}

async function init() {
  const res = await fetch("/public/weights.safetensors", {
    cache: "force-cache",
  });
  const weights = new Uint8Array(await res.arrayBuffer());
  model = new wasm.WasmModel(weights);
  newImagePredict();
}

async function predict() {
  if (imageBytes === null) {
    alert("cannot predict without image");
    return;
  }
  let start = performance.now();
  let probs = model.predict_image(imageBytes);
  let timeElapsed = performance.now() - start;

  setInferenceTime(timeElapsed.toFixed(2));
  setupChart(probs);
}

function setInferenceTime(time) {
  document.getElementById("inference").innerHTML = `Inference time: ${time}ms`;
}

function setupChart(probabilities) {
  if (chart !== null) {
    chart.destroy();
  }

  const ctx = document.getElementById("chart");
  const height = document.getElementById("content").clientHeight;
  // ctx.style.maxHeight = (height - 100) + "px";

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
      backgroundColor: "rgba(202, 138, 4, 0.8)",
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 0,
          max: 1,
          // display: false,
        },
        x: {
          ticks: {
            autoSkip: false,
          },
        },
      },
      plugins: {
        legend: {
          display: false,
        },
      },
      animation: {
        duration: 0,
      },
    },
  });
}

function main() {
  let randomButton = document.getElementById("random");
  init();

  randomButton.addEventListener("click", () => {
    newImagePredict();
  });
}

document.addEventListener("DOMContentLoaded", main);
// window.onload = main

if (document.readyState !== "loading") {
  main();
}
