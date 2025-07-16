# HTHTA-ViT++: An Explainable and Efficient Vision Transformer with Hierarchical GRU-Guided Token Attention

This repository contains the official implementation of **HTHTA-ViT++**, a novel Vision Transformer architecture that combines hierarchical token attention with bidirectional GRU-based sequence modeling for improved efficiency and interpretability.

## 🎯 Key Features

- **Hierarchical Token Attention**: Multi-level aggregation of multiscale feature representations
- **Bidirectional GRU Integration**: Enhanced sequential pattern modeling between spatial tokens
- **Interpretable Attention**: Visualizable attention maps with Focused Attention Percentage (FAP) metric
- **Efficient Architecture**: 13% reduction in FLOPs compared to ViT-B/16 while maintaining performance
- **Strong Performance**: State-of-the-art results on CIFAR-10/100, Tiny-ImageNet, and Intel Image Classification

## 📊 Performance Results

| Dataset | HTHTA-ViT++ | ViT-B/16 | ConvNeXt-B | Improvement |
|---------|-------------|----------|------------|-------------|
| Intel Image Classification | 97.9% | 93.1% | 95.8% | +2.1% |
| CIFAR-10 | 98.7% | 96.5% | 98.1% | +0.6% |
| CIFAR-100 | 93.3% | 84.6% | 89.2% | +4.1% |
| Tiny-ImageNet | 88.9% | 76.8% | 82.6% | +6.3% |

**Interpretability**: FAP score of 78.3% (vs. 53.7% for ViT-B/16)

## 🏗️ Architecture Overview

HTHTA-ViT++ consists of three main components:

1. **Bidirectional GRU Token Sequencer**: Models sequential dependencies among patch tokens
2. **Multi-Head Attention Pooling**: Selectively focuses on class-relevant regions with interpretability
3. **Hierarchical CLS-Token Fusion**: Adaptively combines global and local representations

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hthta-vit-plus-plus.git
cd hthta-vit-plus-plus

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train on CIFAR-100
python train.py --config configs/cifar100.yaml

# Train on Tiny-ImageNet
python train.py --config configs/tiny_imagenet.yaml

# Train with custom parameters
python train.py --config configs/custom.yaml --batch_size 32 --learning_rate 2e-5
```

### Evaluation

```bash
# Evaluate trained model
python evaluate.py --model_path checkpoints/best_model.pth --dataset cifar100

# Generate attention visualizations
python visualize_attention.py --model_path checkpoints/best_model.pth --image_path sample_images/
```


## 🔧 Configuration

All model and training configurations are managed through YAML files in the `configs/` directory. Key parameters include:

- **Model Architecture**: Embedding dimensions, number of heads, GRU layers
- **Training**: Learning rate, batch size, optimizer settings
- **Data**: Dataset paths, augmentation parameters
- **Evaluation**: Metrics, visualization settings

Example configuration:
```yaml
model:
  embed_dim: 768
  num_heads: 12
  num_layers: 12
  gru_layers: 2
  attention_heads: 8
  fusion_params:
    gamma: 0.5
    beta: 0.3

training:
  epochs: 30
  batch_size: 32
  learning_rate: 2e-5
  weight_decay: 0.01
  warmup_steps: 500
```

## 📈 Datasets

The model supports the following datasets:

1. **CIFAR-10/100**: Automatic download and preprocessing
2. **Tiny-ImageNet**: Download from [Stanford CS231n](https://tiny-imagenet.herokuapp.com/)
3. **Intel Image Classification**: Download from [Kaggle](https://www.kaggle.com/puneet6060/intel-image-classification)

### Data Preparation

```bash
# Download and prepare datasets
python scripts/download_datasets.py --datasets cifar10 cifar100 tiny_imagenet
python scripts/preprocess_data.py --dataset_path data/raw --output_path data/processed
```

## 🎨 Attention Visualization

HTHTA-ViT++ provides interpretable attention maps through the FAP (Focused Attention Percentage) metric:

```python
from src.evaluation.visualizer import AttentionVisualizer

visualizer = AttentionVisualizer(model, device='cuda')
attention_maps = visualizer.generate_attention_maps(images)
fap_score = visualizer.calculate_fap(attention_maps, ground_truth_regions)
```

## 🧪 Experiments and Ablation Studies

Run ablation studies to understand component contributions:

```bash
# Run full ablation study
python experiments/ablation_study.py --components bigru attention_pooling cls_fusion

# Compare with baseline models
python experiments/baseline_comparison.py --models vit_base deit_base swin_base convnext_base
```

## 📊 Monitoring and Logging

Training progress can be monitored using:

- **TensorBoard**: `tensorboard --logdir logs/tensorboard`
- **Weights & Biases**: Configure API key in `configs/logging.yaml`
- **Console logging**: Real-time training metrics

## 🔍 Testing

Run the test suite to ensure everything works correctly:

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_model.py -v
pytest tests/test_data.py -v
pytest tests/test_training.py -v
```

## 🚀 Deployment

For production deployment:

```bash
# Export model to ONNX
python scripts/export_onnx.py --model_path checkpoints/best_model.pth

# Quantize model for edge deployment
python scripts/quantize_model.py --model_path checkpoints/best_model.pth --quantization_method dynamic
```

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@article{khurmi2024hthta,
  title={HTHTA-ViT++: An Explainable and Efficient Vision Transformer with Hierarchical GRU-Guided Token Attention},
  author={Khurmi, Simer and Yadav, Naincy and Sharma, Prisha and Arora, Vidushi and Bharti, Surbhi and Kumar, Ashwini},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For questions and support:

- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions for general questions
- **Email**: Contact the authors at the emails provided in the paper

## 🙏 Acknowledgments

- Vision Transformer (ViT) implementation inspired by [timm](https://github.com/rwightman/pytorch-image-models)
- Attention visualization based on [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
- Thanks to the PyTorch team for the excellent deep learning framework

## 📋 Changelog

### v1.0.0 (2024-XX-XX)
- Initial release of HTHTA-ViT++
- Complete implementation with all components
- Comprehensive evaluation on four datasets
- Attention visualization and FAP metric
- Documentation and examples

---

**Note**: This is a research implementation. For production use, please ensure thorough testing and validation for your specific use case.
