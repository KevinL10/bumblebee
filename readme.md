# Bumblebee 


todo: 
- fix old dependencies for webpack
- add rust build stage to vercel (currently only runs `npm install && npm run build`)

model implementation

- [x] load cifar-10 dataset and update training process (batching)
- [x] add learnable start token `[class]` & classify
- [x] add layernorm & multi-head attention
- [x] add layernorm & mlp & repeat
- [ ] benchmarking
- [ ] add dropout for training 

compilation:
- [x] save & load weights 
- [ ] compile to wasm (webgpu)
- [ ] add boilerplate index.html & js loader
