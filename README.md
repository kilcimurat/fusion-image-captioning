# Fusion Image Captioning

This is the official codebase for paper: Fusion of High-Level Visual Attributes for Image Captioning [https://dergipark.org.tr/tr/download/article-file/3345449].


```
@article{kilci2023fusion,
  title={Fusion of High-Level Visual Attributes for Image Captioning},
  author={Kılcı, Murat and Cayli, Özkan and Kılıç, Volkan},
  journal={Avrupa Bilim ve Teknoloji Dergisi},
  number={52},
  pages={161--168},
  year={2023}
}
```
## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Result](#result)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## Introduction

The goal of this project is to generate descriptive captions for images by leveraging high-level visual attributes. The approach is based on the paper "Fusion of High-Level Visual Attributes for Image Captioning," which explores the fusion of different visual features to improve the quality of generated captions.

## Features

- Image feature extraction using pre-trained deep learning models
- Caption generation using a fusion model that combines visual attributes
- Pre-processing and post-processing of images and captions
- Evaluation metrics for caption quality

## Requirements

This project requires the following libraries:

- Python 3.x
- PyTorch with CUDA support
- Torchvision
- Torchaudio
- Torchtext
- Timm
- Tqdm
- Torchinfo
- Torch-TB-Profiler
- TensorBoard
- PyCOCOTools
- pycocoevalcap
- SpaCy and English language model
- OpenJDK 8 (for SPICE Metric)

## Installation

To get started with this project, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/kilcimurat/fusion-image-captioning.git
    cd fusion-image-captioning
    ```

2. **Create and activate a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required libraries:**

Install the libraries and dependencies in environment.txt

## Usage

You can start generating captions for images by running the main script. For example:

```bash
python main.py --image_path path_to_your_image.jpg
```
This will process the image at `path_to_your_image.jpg` and generate a descriptive caption based on the trained model.

## Result
Our model was evaluated on the VizWiz and MSCOCO Captions datasets. Below are some of the results achieved by our model.

### Examples
![resim](https://github.com/kilcimurat/fusion-image-captioning/blob/main/Screenshot%20from%202024-06-29%2011-21-39.png)


### Result Graph
Our experimental evaluations demonstrated the effectiveness of fusing high-level visual attributes. Below are the result graphs from our study:
![resim](https://github.com/kilcimurat/fusion-image-captioning/blob/main/graph.png)


## Project Structure

- `data/`: Contains datasets and pre-trained models.
- `scripts/`: Contains various scripts for training, testing, and evaluation.
- `models/`: Contains model definitions and architectures.
- `utils/`: Contains utility functions.
- `main.py`: The main script to run the captioning process.
- `environment.txt`: A list of required Python libraries.
- `README.md`: This readme file.


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


