pub mod model;

use model::Model;

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, Var};
use candle_datasets::vision::cifar::load_dir;
use candle_nn::ops::softmax;
use candle_nn::{
    conv2d, linear, loss::cross_entropy, AdamW, Conv2d, Conv2dConfig, Linear, Optimizer,
    ParamsAdamW, VarBuilder, VarMap,
};
use candle_nn::{
    embedding, layer_norm, seq, Activation, Embedding, LayerNorm, LayerNormConfig, Sequential,
};

use rand::seq::SliceRandom;
use rand::thread_rng;

const NUM_CLASSES: usize = 10;
const N_BLOCKS: usize = 2;
const PATCH_SIZE: usize = 4;
const HIDDEN_SIZE: usize = 32;
const INTERMEDIATE_SIZE: usize = HIDDEN_SIZE * 4;
const LEARNING_RATE: f64 = 3e-4;
const BATCH_SIZE: usize = 256;
const EPOCH: usize = 50;

// NUM_HEADS * HEAD_SIZE = HIDDEN_SIZE
const NUM_HEADS: usize = 4;

const NUM_PATCHES: usize = (32 * 32) / (PATCH_SIZE * PATCH_SIZE);

fn main() -> Result<()> {
    // load dataset
    const TRAIN_SZ: usize = 2000;
    const TEST_SZ: usize = 1000;
    let device = Device::new_cuda(0)?;

    let dataset = load_dir("data/cifar-10")?;
    let train_images = dataset
        .train_images
        .i((..TRAIN_SZ, .., .., ..))?
        .to_device(&device)?;
    let train_labels = dataset
        .train_labels
        .i(..TRAIN_SZ)?
        .to_dtype(DType::U32)?
        .to_device(&device)?;

    let test_images = dataset
        .test_images
        .i((..TEST_SZ, .., .., ..))?
        .to_device(&device)?;
    let test_labels = dataset
        .test_labels
        .i(..TEST_SZ)?
        .to_dtype(DType::U32)?
        .to_device(&device)?;

    println!("Finished loading dataset");
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Model::new(vb.clone())?;

    varmap.load("weights.safetensors")?;


    // training
    let params = candle_nn::ParamsAdamW {
        lr: LEARNING_RATE,
        ..Default::default()
    };
    let mut optimizer = AdamW::new(varmap.all_vars(), params)?;
    let n_batches = train_images.dim(0)? / BATCH_SIZE;
    let mut batch_idx = (0..n_batches).collect::<Vec<usize>>();

    for epoch in 1..EPOCH {
        batch_idx.shuffle(&mut thread_rng());
        let mut total_loss = 0f32;

        for idx in batch_idx.iter() {
            let logits = model.forward(&train_images.narrow(0, idx * BATCH_SIZE, BATCH_SIZE)?)?;
            let loss = cross_entropy(
                &logits,
                &train_labels.narrow(0, idx * BATCH_SIZE, BATCH_SIZE)?,
            )?;
            optimizer.backward_step(&loss)?;
            total_loss += loss.to_vec0::<f32>()?;
        }

        if epoch % 5 == 0 {
            let logits = model.forward(&test_images)?;

            println!("{:?}", logits.shape());
            let n_correct = logits
                .argmax(1)?
                .eq(&test_labels)?
                .to_dtype(DType::F32)?
                .sum_all()?
                .to_scalar::<f32>()?;

            let accuracy = n_correct / (test_labels.dim(0)? as f32);
            println!(
                "epoch {epoch}/{EPOCH}. Training: {total_loss};  Validation: {0}%",
                accuracy * 100f32
            );
        }
    }

    varmap.save("weights.safetensors")?;

    Ok(())
}
