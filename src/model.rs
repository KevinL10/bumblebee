use candle_core::{Device, IndexOp, Module, Result, Tensor, DType};
use candle_nn::ops::softmax;
use candle_nn::{conv2d, linear, Conv2d, Conv2dConfig, Linear, VarBuilder, VarMap};
use candle_nn::{embedding, layer_norm, linear_no_bias, Embedding, LayerNorm, LayerNormConfig};

use wasm_bindgen::prelude::*;


// use serde::Deserialize;

const NUM_CLASSES: usize = 10;
const N_BLOCKS: usize = 2;
const PATCH_SIZE: usize = 4;
const HIDDEN_SIZE: usize = 32;
const INTERMEDIATE_SIZE: usize = HIDDEN_SIZE * 4;

// NUM_HEADS * HEAD_SIZE = HIDDEN_SIZE
const NUM_HEADS: usize = 4;

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

#[derive(Debug)]
struct AttentionHead {
    key: Linear,
    query: Linear,
    value: Linear,
}

impl AttentionHead {
    fn new(vb: VarBuilder, head_size: usize) -> Result<Self> {
        let key = linear_no_bias(HIDDEN_SIZE, head_size, vb.pp("key"))?;
        let query = linear_no_bias(HIDDEN_SIZE, head_size, vb.pp("key"))?;
        let value = linear_no_bias(HIDDEN_SIZE, head_size, vb.pp("key"))?;

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

#[derive(Debug)]
struct MultiAttentionHead {
    heads: Vec<AttentionHead>,
}

impl MultiAttentionHead {
    fn new(vb: VarBuilder) -> Result<Self> {
        // take hidden size -> hidden size
        let head_size = HIDDEN_SIZE / NUM_HEADS;
        let mut heads: Vec<AttentionHead> = Vec::new();
        for i in 0..NUM_HEADS {
            heads.push(AttentionHead::new(vb.pp(format!("head-{i}")), head_size)?)
        }

        Ok(Self { heads })
    }
}

impl Module for MultiAttentionHead {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let xs = self
            .heads
            .iter()
            .map(|head| head.forward(x).unwrap())
            .collect::<Vec<Tensor>>();
        Tensor::cat(&xs, 2)
    }
}

#[derive(Debug)]
struct Block {
    attention: MultiAttentionHead,
    mlp: MLP,
    ln1: LayerNorm,
}

impl Block {
    fn new(vb: VarBuilder) -> Result<Self> {
        let mlp = MLP::new(vb.pp("mlp"))?;
        let attention = MultiAttentionHead::new(vb.pp("attention"))?;
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

#[derive(Debug)]
pub struct Model {
    patch_embedding: Conv2d,
    pos_embedding: Embedding,
    cls_token: Tensor,
    classifier: Linear,
    blocks: Vec<Block>,
}

impl Model {
    pub fn info(&self) -> String {
        format!("Model: {:?}", self)
    }


    pub fn new(vb: VarBuilder) -> Result<Model> {
        let conv_cfg = Conv2dConfig {
            stride: PATCH_SIZE,
            ..Default::default()
        };

        let cls_token = vb.get((1, HIDDEN_SIZE), "cls_token")?;

        // # of patches = 64
        let pos_embedding = embedding(NUM_PATCHES + 1, HIDDEN_SIZE, vb.pp("pos_embedding"))?;
        let patch_embedding = conv2d(
            3,
            HIDDEN_SIZE,
            PATCH_SIZE,
            conv_cfg,
            vb.pp("patch_embedding"),
        )?;
        let classifier = linear(HIDDEN_SIZE, NUM_CLASSES, vb.pp("classifier"))?;

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

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
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


