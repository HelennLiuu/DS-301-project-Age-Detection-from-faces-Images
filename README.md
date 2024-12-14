# DS-301-project-Age-Detection-from-faces-Images

# 1. Motivation

We aim to answer the question: Can we accurately classify human faces into specific age groups – namely "YOUNG," "MIDDLE," and "OLD" – using deep learning models? Age recognition has practical applications in various fields, such as demographic analysis, targeted marketing, security systems, and even healthcare, where age-related assessments can assist in early diagnosis of age-associated diseases.
Age classification presents an exciting challenge due to the complexity of facial aging patterns, which vary by individual, gender, ethnicity, and other factors. Current advances in deep learning have demonstrated impressive results in tasks like image recognition, but accurate age prediction from facial images remains difficult due to subtle differences in facial features across age groups.

# 2. Dataset
The dataset consists of 19,906 facial images in various resolutions (ranging from 8x11 to over 724x500 pixels), paired with a CSV file, "train.csv," which contains two columns: "ID" (the image filenames) and "Class" (the corresponding age group: "YOUNG," "MIDDLE," or "OLD"). This dataset is suitable for our deep learning model’s task of age classification because it provides a diverse range of face images that represent different age groups. The variety in resolution and file size may present challenges in data preprocessing, but deep learning models are well-suited to handle such variability.

(./dataset_distribution_plot.png)



