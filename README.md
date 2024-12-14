# DS-301-project-Age-Detection-from-faces-Images

# 1. Motivation

We aim to answer the question: Can we accurately classify human faces into specific age groups – namely "YOUNG," "MIDDLE," and "OLD" – using deep learning models? Age recognition has practical applications in various fields, such as demographic analysis, targeted marketing, security systems, and even healthcare, where age-related assessments can assist in early diagnosis of age-associated diseases.
Age classification presents an exciting challenge due to the complexity of facial aging patterns, which vary by individual, gender, ethnicity, and other factors. Current advances in deep learning have demonstrated impressive results in tasks like image recognition, but accurate age prediction from facial images remains difficult due to subtle differences in facial features across age groups.

# 2. Dataset
The dataset consists of 19,906 facial images in various resolutions (ranging from 8x11 to over 724x500 pixels), paired with a CSV file, "train.csv," which contains two columns: "ID" (the image filenames) and "Class" (the corresponding age group: "YOUNG," "MIDDLE," or "OLD"). This dataset is suitable for our deep learning model’s task of age classification because it provides a diverse range of face images that represent different age groups. The variety in resolution and file size may present challenges in data preprocessing, but deep learning models are well-suited to handle such variability.

![Dataset Distribution](dataset%20distribution%20plot.png)

You can download dataset here:
https://www.kaggle.com/datasets/arashnic/faces-age-detection-dataset

# 3. Methodology
## Convolutional Neural Networks (CNNs)

Data Preprocessing: The facial images were resized to 128x128 pixels to ensure
consistent input dimensions, converted to RGB, and normalized to the [0, 1] range. Age
group labels (YOUNG, MIDDLE, OLD) were encoded as integers (0, 1, 2) to handle age
classification as a categorical task. The dataset was divided into training and validation
sets with an 80-20 split to evaluate the model’s generalization.

Model Architecture: The CNN model consists of four convolutional blocks, each
followed by max-pooling layers to capture spatial hierarchies and reduce dimensionality.
The convolutional layers start with 32 filters and increase to 128, followed by a flattening
layer, a dense layer with 512 units, and an output layer with three units using softmax
activation for multi-class classification.

Model Compilation and Training: The model was compiled using Sparse Categorical
Crossentropy as the loss function and the Adam optimizer for efficient gradient updates.
A checkpoint callback saved the best-performing model based on validation accuracy,
mitigating overfitting. Training was conducted over 10 epochs with a batch size of 32.
Evaluation and Analysis: Training and validation accuracy and loss metrics were
tracked across epochs, and plotted to assess model learning patterns, convergence, and
potential overfitting.

## Transfer Learning (ResNet and MobileNet)

Data Preparation and Preprocessing: Images were resized to 128x128 pixels,
converted to RGB, and normalized. For ResNet50, pixel values were scaled between -1
and 1, while MobileNetV2 utilized its own preprocessing function. Age labels were
encoded as integers (0, 1, 2).

Model Architecture: Pre-trained models ResNet50 and MobileNetV2 were used for
transfer learning. Initially, the base model layers were frozen to leverage pretrained
features from ImageNet. A custom classification head was added with a
GlobalAveragePooling2D layer, a dropout layer (rate of 0.2), and a dense layer with
softmax activation.

Training and Fine-Tuning: Each model was trained in a feature extraction phase with
a learning rate of 0.0001. The last 10 layers were unfrozen for fine-tuning, and a reduced
learning rate of 1e-5 was applied for stability. Metrics were plotted to compare the efficacy
of the fine-tuning phase.

## Multi-Task Learning and Attention Mechanisms

Data Preprocessing: Images were resized to 160x160 pixels, normalized to [0, 1], and
labels were one-hot encoded. The dataset was split into training and validation sets.

Model Architecture: The model used four convolutional blocks (Conv2D,
BatchNormalization, MaxPooling2D, and SpatialDropout2D) and integrated a Channel
Attention layer. Average and max pooling generated attention maps to focus on
important facial regions. GlobalAveragePooling2D was used to reduce spatial dimensions,
followed by a dense layer with ReLU and Dropout, and a final softmax layer.

Training and Evaluation: An exponentially decaying learning rate (starting at 0.001)
and Adam optimizer were applied. After 15 epochs, accuracy and loss on validation data
were visualized, with training history plots to analyze performance.


