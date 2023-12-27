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

// use serde::Deserialize;

const NUM_CLASSES: usize = 10;
const N_BLOCKS: usize = 2;
const PATCH_SIZE: usize = 4;
const HIDDEN_SIZE: usize = 32;
const INTERMEDIATE_SIZE: usize = HIDDEN_SIZE * 4;
const LEARNING_RATE: f64 = 3e-4;
const BATCH_SIZE: usize = 256;

const HEAD_SIZE: usize = 32;

const NUM_PATCHES: usize = (32 * 32) / (PATCH_SIZE * PATCH_SIZE);

#[derive(Debug)]

struct MLP {
    layer1: Linear,
    // gelu: Activation,
    layer2: Linear,
}

impl MLP {
    fn new(vb: VarBuilder) -> Result<Self> {
        let layer1 = linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, vb.pp("mlp_l1"))?;
        let layer2 = linear(INTERMEDIATE_SIZE, HIDDEN_SIZE, vb.pp("mlp_l2"))?;

        Ok(Self { layer1, layer2 })
    }
}

// MLP used in each block; takes (b_size, n_patches, embedding) --> (b_size, n_patches, embedding')
// use dropout for training?
impl Module for MLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.apply(&self.layer1)?.gelu()?.apply(&self.layer2)
    }
}

struct AttentionHead {
    key: Linear,
    query: Linear,
    value: Linear,
}

impl AttentionHead {
    fn new(vb: VarBuilder) -> Result<Self> {
        let key = linear(HIDDEN_SIZE, HEAD_SIZE, vb.pp("key"))?;
        let query = linear(HIDDEN_SIZE, HEAD_SIZE, vb.pp("key"))?;
        let value = linear(HIDDEN_SIZE, HEAD_SIZE, vb.pp("key"))?;

        Ok(Self { key, query, value })
    }
}

impl Module for AttentionHead {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let k = x.apply(&self.key)?; // (B, n_patches, HEAD_SIZE)
        let q = x.apply(&self.query)?;
        let scale: f64 = (HIDDEN_SIZE as f64).powf(-0.5);

        let weights = (q.matmul(&k.transpose(1, 2)?)? * scale)?;
        let weights = softmax(&weights, 1)?;

        // (B, n_patches, n_patches) * (B, n_patches, HIDDEN_SIZE)
        let v = x.apply(&self.value)?;
        weights.matmul(&v)
    }
}

struct Block {
    attention: AttentionHead,
    mlp: MLP,
    ln1: LayerNorm,
}

impl Block {
    fn new(vb: VarBuilder) -> Result<Self> {
        let mlp = MLP::new(vb.pp("mlp"))?;
        let attention = AttentionHead::new(vb.pp("attention"))?;
        let ln1 = layer_norm(
            HIDDEN_SIZE,
            LayerNormConfig {
                ..Default::default()
            },
            vb.pp("ln1"),
        )?;
        Ok(Self {
            mlp,
            attention,
            ln1,
        })
    }
}

impl Module for Block {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // layer norm -> multihead   (and add original x for residual connection)
        let x = (x + x.apply(&self.ln1)?.apply(&self.attention))?;

        // lyer norm -> mlp
        &x + &x.apply(&self.ln1)?.apply(&self.mlp)?
    }
}

struct Model {
    patch_embedding: Conv2d,
    pos_embedding: Embedding,
    cls_token: Tensor,
    classifier: Linear,
    blocks: Vec<Block>,
}

impl Model {
    fn new(vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            stride: PATCH_SIZE,
            ..Default::default()
        };

        let cls_token = vb.get((1, HIDDEN_SIZE), "cls_token")?;

        // # of patches = 64
        let pos_embedding = embedding(64 + 1, HIDDEN_SIZE, vb.pp("pos_embedding"))?;
        let patch_embedding = conv2d(
            3,
            HIDDEN_SIZE,
            PATCH_SIZE,
            conv_cfg,
            vb.pp("patch_embedding"),
        )?;
        let classifier = linear(HIDDEN_SIZE, 10, vb.pp("classifier"))?;

        let mut blocks: Vec<Block> = Vec::new();

        for i in 0..N_BLOCKS {
            blocks.push(Block::new(vb.pp(format!("block-{i}")))?)
        }

        Ok(Self {
            patch_embedding,
            pos_embedding,
            cls_token,
            classifier,
            blocks,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // the current batch size (could be different from BATCH_SIZE when testing)
        let (b_size, _, _, _) = x.dims4()?;

        // apply patch embedding
        let mut x = self.patch_embedding.forward(x)?;
        x = x.flatten(2, 3)?.transpose(1, 2)?;

        // add [class] token
        let cls = self.cls_token.expand((b_size, HIDDEN_SIZE))?.unsqueeze(1)?;
        x = Tensor::cat(&[&x, &cls], 1)?;

        // add pos embedding
        let pos_idx = (0..(NUM_PATCHES + 1) as u32).collect::<Vec<u32>>();
        let pos_tensor = Tensor::new(&pos_idx[..], &Device::new_cuda(0)?)?;
        let pos_embedding = self.pos_embedding.forward(&pos_tensor)?;

        x = x.broadcast_add(&pos_embedding)?;
        // println!("{:?}", x.shape());

        // encode with transformer
        for block in self.blocks.iter() {
            x = x.apply(block)?;
        }

        // get [class] embedding
        let cls_embed = x.i((.., 0, ..))?;
        let logits = self.classifier.forward(&cls_embed)?;
        // println!("Applied classification layer: {:?}", logits.shape());
        Ok(logits)
    }
}

fn main() -> Result<()> {
    // load dataset
    // const SZ: usize = 1000;
    let device = Device::new_cuda(0)?;

    let dataset = load_dir("data/cifar-10")?;
    let train_images = dataset
        .train_images
        // .i((..SZ, .., .., ..))?
        .to_device(&device)?;
    let train_labels = dataset
        .train_labels
        // .i(..SZ)?
        .to_dtype(DType::U32)?
        .to_device(&device)?;

    let test_images = dataset
        .test_images
        // .i((..SZ, .., .., ..))?
        .to_device(&device)?;
    let test_labels = dataset
        .test_labels
        // .i(..SZ)?
        .to_dtype(DType::U32)?
        .to_device(&device)?;

    println!("Finished loading dataset");
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Model::new(vb)?;

    // training
    let params = candle_nn::ParamsAdamW {
        lr: LEARNING_RATE,
        ..Default::default()
    };
    let mut optimizer = AdamW::new(varmap.all_vars(), params)?;
    let n_batches = train_images.dim(0)? / BATCH_SIZE;
    let mut batch_idx = (0..n_batches).collect::<Vec<usize>>();

    for epoch in 1..100 {
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
            println!("epoch {epoch}/100. Training: {total_loss};  Validation: {0}%", accuracy * 100f32);
        }
    }

    // println!("{logits}");
    Ok(())
}
