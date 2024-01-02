cargo build 
npm install --prefix site


wasm-pack build --out-dir site/pkg
npm run build --prefix site