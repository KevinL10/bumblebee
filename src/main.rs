use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{conv2d, linear, Conv2d, Conv2dConfig, Optimizer, Linear, VarBuilder, AdamW, ParamsAdamW, VarMap, loss::cross_entropy};
// use serde::Deserialize;

const NUM_CLASSES: usize = 10;
const DEVICE: Device = Device::Cpu;
const PATCH_SIZE: usize = 4;
const D: usize = 32;
const LEARNING_RATE: f64 = 1e-3;
#[derive(Debug)]

struct Model {
    patch_embedding: Conv2d,
    classifier: Linear,
}

impl Model {
    fn new(vs: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            stride: PATCH_SIZE,
            ..Default::default()
        };

        let patch_embedding = conv2d(3, D, PATCH_SIZE, conv_cfg, vs.pp("embedding"))?;
        let classifier = linear(D, 10, vs.pp("classifier"))?;
        Ok(Self {
            patch_embedding,
            classifier
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.patch_embedding.forward(x)?;
        x = x.flatten(2, 3)?.transpose(1, 2)?;

        // println!("Applied patch embedding: {:?}", x.shape());
        let cls_embed = x.i((.., 0, ..))?;

        let logits = self.classifier.forward(&cls_embed)?;
        // println!("Applied classification layer: {:?}", logits.shape());
        Ok(logits)
    }
}

fn main() -> Result<()> {
    let image = Tensor::randn(0f32, 1., (4, 3, 32, 32), &Device::Cpu)?;
    let targets: Vec<i64> = vec![3, 9, 2, 3];
    let targets = Tensor::from_vec(targets, (4, ),&Device::Cpu)?;

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

    let model = Model::new(vs)?;

    let params = candle_nn::ParamsAdamW {
        lr: LEARNING_RATE,
        ..Default::default()
    };
    let mut optimizer = AdamW::new(varmap.all_vars(), params)?;


    for epoch in 1.. 3900 {
        let logits = model.forward(&image)?;
        let loss = cross_entropy(&logits, &targets)?;
        optimizer.backward_step(&loss)?;

        let total_loss  = loss.to_vec0::<f32>()?; 

        if epoch % 100 == 0{
            println!("loss: {total_loss}");
        }
    }

    // println!("{logits}");
    Ok(())
}
