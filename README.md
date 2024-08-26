# 🌾 Rice Image Classification Using CNN

<img src="readme_image.jpg" width="1000" height="466">

## Project Overview
Artificial Intelligence (AI) is revolutionizing agriculture by automating tasks such as rice variety classification, which is crucial for quality and economic reasons, especially in regions like Turkey. This project employs a Convolutional Neural Network (CNN) to automate the classification of rice grains based on visual features like shape, color, and texture. By replacing labor-intensive and error-prone manual methods with a machine learning model, the project aims to enhance efficiency, accuracy, and reliability in rice production, demonstrating AI’s potential to transform traditional agricultural practices.

## Datasets
The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset). It includes the following classes:
- Arborio, Basmati, Ipsala, Jasmine and Karacadag.
- each class contains 15000 images and total it contains 75000 images with 12 morphological, 4 shape and 90 color features. 

## Tools and Technologies Used
- **Data Analysis:** Python (Pandas,Numpy)
- **Machine Learning:** TensorFlow, Keras(for CNN implementation) 
- **Visualization:** Matplotlib, Seaborn
- **Version Control:** Git, GitHub

## Installation and Usage
**Prerequisites**
Ensure you have Python installed on your machine. You will also need to install the required libraries:

```bash
# Install dependencies
pip install -r requirements.txt
```
**Running the Project**
```bash
# Clone the repository
git clone https://github.com/puni-ram48/Health-Insurance-Charges-Prediction.git
```

[**Project Analysis Report**](analysis_report.ipynb): Final report containing data analysis and visualizations and the model development .

[**requirements.txt**](requirements.txt): List of required python libraries.

## Model Development and Evaluation

**1. Data Collection and Preprocessing**
- Utilized 75,000 images from Kaggle, covering five rice varieties to ensure balanced representation.

**2. Data Shuffling and Splitting**
- Shuffled and split the dataset into training (70%), validation (15%), and testing (15%) sets to avoid bias and ensure accurate evaluation.

**3. Data Normalization and Augmentation**
- Normalized pixel values and applied data augmentation techniques (rotations, flips, zooms) using ImageDataGenerator to improve model robustness.
  
**4. Model Development**
- The CNN model features convolutional layers for feature extraction, max-pooling layers for dimensionality reduction, and fully connected layers for classification, with ReLU activations in hidden layers and softmax in the output layer.
- It is compiled with categorical cross-entropy loss and the Adam optimizer for efficient training.
    
**5. Model Evaluation**:
The model is evaluated using accuracy metrics.
  - Training Accuracy: 99.11% with a training loss of 0.0275
  - Validation Accuracy: 98.98% with a validation loss of 0.0335
  - Test Accuracy: 99.00% with a test loss of 0.0275

## Contributing
We welcome contributions to this project! If you would like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Make your changes and commit them (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Create a new Pull Request.

Please ensure your code is well-documented.

## Authors and Acknowledgment
This project was initiated and completed by Puneetha Dharmapura Shrirama.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
