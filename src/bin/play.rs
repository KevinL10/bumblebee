use candle_datasets::vision::Dataset;
use bumblebee::model::Model;

use candle_core::{DType, Device, Tensor, IndexOp, Result};
use candle_datasets::vision::cifar::load_dir;

use candle_nn::{loss::cross_entropy, AdamW, Optimizer, VarBuilder, VarMap};

use rand::seq::SliceRandom;
use rand::thread_rng;

const LEARNING_RATE: f64 = 3e-4;
const BATCH_SIZE: usize = 256;
const EPOCH: usize = 300;


use image::{ImageBuffer, Rgb};

fn main() -> Result<()> {
    // Assuming `tensor` is your (3, 32, 32) tensor containing image data

    // Create a new ImageBuffer
    let dataset = load_dir("data/cifar-10")?;
    let train_images = dataset
        .train_images
        .i((1, .., .., ..))?; 
    let train_labels = dataset
        .train_labels
        .i((1, ))?;

    println!("{:?}", train_images.shape());
    println!("{:?}", train_labels.shape());
    println!("{:?}", train_labels);


    let mut imgbuf = ImageBuffer::new(32, 32);

    for (y, x, pixel) in imgbuf.enumerate_pixels_mut() {
        // Retrieve RGB values from the tensor
        // For example, assuming the tensor is in (channel, width, height) format
        // let r = train_images.i((0,x as usize,y as usize))?.t;
        let s = train_images.i((0,x as usize,y as usize))?;
        // println!("{:?}", s.shape());
        let r = train_images.i((0,x as usize,y as usize))?.to_scalar::<f32>()? ;
        let g = train_images.i((1,x as usize,y as usize))?.to_scalar::<f32>()? ;
        let b = train_images.i((2,x as usize,y as usize))?.to_scalar::<f32>()? ;

        // println!("R: {}, G: {}, B: {}", r, g, b);

        // Assign the pixel. ImageBuffer uses (x, y) coordinate system.
        *pixel = Rgb([(r * 256f32) as u8, (g * 256f32) as u8, (b * 256f32) as u8]);
    }

    // Save or display the image
    imgbuf.save("output.png").unwrap();
    Ok(())
}