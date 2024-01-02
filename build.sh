# Compile Rust to WASM
cargo build 
wasm-pack build --out-dir site/pkg

# Install package and build site
npm install --prefix site
npm run build --prefix site