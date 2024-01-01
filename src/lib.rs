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

use candle_core::{Device, IndexOp, Module, Tensor, DType};
use candle_nn::ops::softmax;
use candle_nn::{conv2d, linear, Conv2d, Conv2dConfig, Linear, VarBuilder, VarMap};
use candle_nn::{embedding, layer_norm, linear_no_bias, Embedding, LayerNorm, LayerNormConfig};

#[wasm_bindgen]
pub struct WasmModel {
    // model: Model    
    model: Model 
}

#[wasm_bindgen]
impl WasmModel {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WasmModel, JsError> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let model = Model::new(vb.clone())?;


        Ok(Self {
            model: model
        })
    }

    pub fn name(&self) -> String {
        self.model.info()
    }
}