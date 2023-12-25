use crate::model::{train, ModelConfig, TrainingConfig};
use burn::backend::wgpu::AutoGraphicsApi;
use burn::backend::{Autodiff, Wgpu};
use burn::optim::AdamConfig;

mod data;
mod model;

fn main() {
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    train::<MyAutodiffBackend>(
        "/kaggle/working/guide",
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device,
    );
}