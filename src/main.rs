use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{conv2d, linear, Conv2d, Conv2dConfig, Optimizer, Linear, VarBuilder, AdamW, ParamsAdamW, VarMap, loss::cross_entropy};
use candle_datasets::vision::cifar::{load_dir};

use rand::thread_rng;
use rand::seq::SliceRandom;


// use serde::Deserialize;

const NUM_CLASSES: usize = 10;
const DEVICE: Device = Device::Cpu;
const PATCH_SIZE: usize = 4;
const D: usize = 32;
const LEARNING_RATE: f64 = 1e-3;
const BATCH_SIZE: usize = 4;


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
    // load dataset
    // let image = Tensor::randn(0f32, 1., (4, 3, 32, 32), &Device::Cpu)?;


    const SZ: usize = 100;
    let dataset = load_dir("data/cifar-10")?;
    let train_images = dataset.train_images.i((..SZ, .., .., ..))?;
    let train_labels = dataset.train_labels.i(..SZ)?;

    let test_images = dataset.test_images.i((..SZ, .., .., ..))?;
    let test_labels = dataset.test_labels.i(..SZ)?;


    // let targets: Vec<i64> = vec![3, 9, 2, 3];
    // let targets = Tensor::from_vec(targets, (4, ),&Device::Cpu)?;

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

    let model = Model::new(vs)?;

    let params = candle_nn::ParamsAdamW {
        lr: LEARNING_RATE,
        ..Default::default()
    };
    let mut optimizer = AdamW::new(varmap.all_vars(), params)?;


    let n_batches = train_images.dim(0)? / BATCH_SIZE;
    let mut batch_idx = (0..n_batches).collect::<Vec<usize>>();


    for epoch in 1.. 100 {
        batch_idx.shuffle(&mut thread_rng());

        let mut total_loss = 0f32;

        for idx in batch_idx.iter() {
            let logits = model.forward(&train_images.narrow(0, idx * BATCH_SIZE, BATCH_SIZE)?)?;
            let loss = cross_entropy(&logits, &train_labels.narrow(0, idx * BATCH_SIZE, BATCH_SIZE)?)?;
            optimizer.backward_step(&loss)?;

            total_loss  += loss.to_vec0::<f32>()?; 
        }

        if epoch % 5 == 0{
            let logits = model.forward(&test_images)?;
            let loss = cross_entropy(&logits,&test_labels)?.to_vec0::<f32>()?;
            println!("epoch {epoch}/100. Training: {total_loss};  Validation: {loss}");
        }
    }

    // println!("{logits}");
    Ok(())
}
