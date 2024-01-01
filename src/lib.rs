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
#[derive(Debug)]
pub struct WasmModel {
    // model: Model    
    model: Model 
}

#[wasm_bindgen]
impl WasmModel {
    pub fn new() -> Result<WasmModel, JsError> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let model = Model::new(vb.clone())?;


        Ok(Self {
            model: model
        })
    }

    #[wasm_bindgen(constructor)]
    pub fn load(weights: Vec<u8>) -> Result<WasmModel, JsError> {
        
        let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, &Device::Cpu)?;
        let model = Model::new(vb.clone())?;
        

        Ok(Self {
            model: model
        })
    }


    
    // https://github.com/huggingface/candle/blob/main/candle-wasm-examples/blip/src/bin/m.rs#L125C4-L125C 
    pub fn predict_image(&self, image: Vec<u8>) -> Result<Vec<f32>, JsError> {
        let device = &Device::Cpu;
        let img = image::io::Reader::new(std::io::Cursor::new(image))
            .with_guessed_format()?
            .decode()
            .map_err(|e| JsError::new(&e.to_string()))?
            .resize_to_fill(32, 32, image::imageops::FilterType::Triangle);
        let img = img.to_rgb8();
        let data = img.into_raw();
        let data = Tensor::from_vec(data, (32, 32, 3), device)?.permute((2, 0, 1))?;
        let mean =
            Tensor::new(&[0.48145466f32, 0.4578275, 0.40821073], device)?.reshape((3, 1, 1))?;
        let std =
            Tensor::new(&[0.26862954f32, 0.261_302_6, 0.275_777_1], device)?.reshape((3, 1, 1))?;
        
        
        let x = (data.to_dtype(DType::F32)? / 255.)?
            .broadcast_sub(&mean)?
            .broadcast_div(&std)
            .map_err(|e| JsError::new(&e.to_string()))?;
    

        let probs = self.model.predict(&x)?;

        Ok(probs) 
    } 


    pub fn info(&self) -> String {
        format!("{:?}", self.model)
    }
}