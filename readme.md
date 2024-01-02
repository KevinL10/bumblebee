# Bumblebee 

Bumblebee is a small vision transformer that runs locally in the browser with WASM, implemented in Rust using [Candle ML](https://github.com/huggingface/candle) framework.

Bumblebee achieves 42% accuracy on the CIFAR-10 dataset. Training took 3 minutes on an NVIDIA 1060 GPU.


todo: 
- fix old dependencies for webpack
- add rust build stage to vercel (currently only runs `npm install && npm run build`)
- fix document loading with index.html
- add dropout to training 
- refactor code, move model/training params to config class