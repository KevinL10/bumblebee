import * as wasm from "webvit";

let imageBytes = null;
let model = null;
let chart = null;

const images = ["airplane1", "cat1"];

async function fetchRandomImage() {
  const img = images[Math.floor(Math.random() * images.length)];
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
  predict();
}

async function init() {
  const res = await fetch("/public/weights.safetensors", {
    cache: "force-cache",
  });
  const weights = new Uint8Array(await res.arrayBuffer());

  model = new wasm.WasmModel(weights);


  fetchRandomImage();
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
  
  let start = performance.now();

  let probs = model.predict_image(imageBytes);
  let timeElapsed = performance.now() - start;
  console.log(probs);
  console.log(timeElapsed);
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
          max: 1
          // display: false,
        },
        x: {
          ticks: {
          autoSkip: false,

          }
        }
      },
      plugins: {
        legend: {
          display: false,
        },
      },
      animation: {
        duration: 0,
      }
    },
  });
}


function main() {
  let randomButton = document.getElementById("random");
  init();

  randomButton.addEventListener("click", () => {
    fetchRandomImage();
  });
}

document.addEventListener('DOMContentLoaded', main);

if (document.readyState !== 'loading') {
    main();
}