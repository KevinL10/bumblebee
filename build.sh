# Compile Rust to WASM
cargo build 
wasm-pack build --out-dir pkg

# Install package and build site
npm install --prefix site
npm run build --prefix site