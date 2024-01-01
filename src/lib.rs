pub mod model;

use model::Model;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern {
    pub fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet(name: &str) {
    alert(&format!("Bye, {}!", name));
} 


#[wasm_bindgen]
pub struct WasmModel {
    // model: Model    
    name: Model 
}

#[wasm_bindgen]
impl WasmModel {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WasmModel, JsError> {
        Ok(Self {
            name: String::from("WasmModel")
        })
    }

    pub fn name(&self) -> String {
        self.name.clone()
    }
}