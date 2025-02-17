# Smoking Behavior Detection

Cigarette detection in photos and videos continues to be a challenging task in computer vision, having major applications in behavior analysis, public health, and security services. In order to address the inherent challenges of this issue, this research presents a novel approach for offline cigarette detection. 

In this project, we started with a CNN custom build model and then opted to use transfer-
learning solutions to ensure better performance and results.

# Data Exploration and Visualization

Our dataset, "Khan, Ali (2020)," consists of a total of 2,400 raw images.

The dataset contains a total of 2,400 raw images:
• 1,200 images in the smoking (smokers) category
• 1,200 images in the not-smoking (non-smokers) category.

The dataset is organized into directories:
• training_data (1,610 images)
• validation_data (400 images)
• testing_data (400 images).

```
There are 3 directories and 0 images in 'Dataset_New'.
There are 2 directories and 0 images in 'Dataset_New/validation_data'.
There are 0 directories and 200 images in 'Dataset_New/validation_data/notsmoking'.
There are 0 directories and 200 images in 'Dataset_New/validation_data/smoking'.
There are 1 directories and 0 images in 'Dataset_New/testing_data'.
There are 0 directories and 400 images in 'Dataset_New/testing_data/test_pictures'.
There are 2 directories and 0 images in 'Dataset_New/training_data'.
There are 0 directories and 805 images in 'Dataset_New/training_data/notsmoking'.
There are 0 directories and 805 images in 'Dataset_New/training_data/smoking'.
There are 805 images of smokers in the training Dataset_New.
There are 805 images of non-smokers in the training Dataset_New.
```
Training_data and validation_data images are labeled, while the testing_data images are
not labeled. This also decreases the number of images for the model to train on, and
therefore limit its performance.

To evaluate our models, the test images must be labeled. Therefore, the validation_data (400
labeled images) is split into two datasets, and new directories are formed: 40% new_validation
(160 labeled images) and 60% test (240 labeled images).

```
Found 160 images belonging to 2 classes.
Found 240 images belonging to 2 classes.
Training set class indices: {'notsmoking': 0, 'smoking': 1}
New validation set class indices: {'notsmoking': 0, 'smoking': 1}
Test set class indices: {'notsmoking': 0, 'smoking': 1}
```

# Data Augmentation

Due to limited data, we decided to use the data augmentation technique to compensate for the
limited data and low number of training images. We performed various augmentations such as
resizing, scaling, flipping, and shifting.

First, the images in the dataset were resized to a common resolution of 224×224. Then, random
augmentations were applied, including scaling the images by up to a factor of 0.2, rotating the
images by up to 20 degrees, and translating the images horizontally or vertically by up to 0.2.
Additionally, shear-based transformations were applied up to a factor of 0.2. The images were
also slightly rotated between 0 and 20 degrees and zoomed in by up to 0.2.

The augmented images (1610 labeled images) are then added to the original training images,
resulting in a total of 3220 labeled images.

```
Found 3220 images belonging to 2 classes.
```

After Augmentation the data looks as follows:
• training_data (3220 images)
• validation_data (160 images)
• testing_data (240 images)


<div align="center">
  <img src="./figures/augmentation.jpg" alt="Data Augmentation">
</div>


# Simulation Parameters Selection

There were several experiments conducted on three models before and after augmentation to
check the effectiveness of data augmentation. The models experimented on were ResNet50-v2,
EfficientNetB0-v2, and MobileNetV2. The parameters selected for training our model are
provided in the table below.

| Parameter      | Value           |
|-------------- |--------------- |
| Input Size    | 224 × 224      |
| Optimizer     | Adam           |
| Loss          | Binary Cross-entropy |
| Learning Rate | 0.001          |
| Batch Size    | 32             |
| Epochs        | 10             |


# Evaluation Metrics

The models were evaluated for accurate classification of smoking and not-smoking images in the
smoker detection dataset. Additionally, the models were compared on various performance
metrics such as prediction accuracy, precision, recall, specificity, positive predictive value (PPV),
negative predictive value (NPV), false negative rate (FNR), false positive rate (FPR), false
discovery rate (FDR), and F1 score. Confusion matrices were utilized to give a clearer
understanding of the model's performance, highlighting both the correct classifications and the
errors the model makes.

<table style="width: 90%; table-layout: fixed;">
  <tr>
    <td align="center">
      <img src="./figures/smoking.jpg" alt="Smoking" width="300">
    </td>
    <td align="center">
      <img src="./figures/nonsmoking.jpg" alt="After" width="300">
    </td>
  </tr>
  <tr>
    <td align="center" style="background-color: #333; color: white; padding: 10px; border-radius: 5px;">
      <strong>Smoking</strong>
    </td>
    <td align="center" style="background-color: #333; color: white; padding: 10px; border-radius: 5px;">
      <strong>Non-smoking</strong> 
    </td>
  </tr>
</table>

# Experiments

In general, data augmentation had a positive impact on the performance of the machine learning models. By increasing the variety and amount of training data, augmentation improved model accuracy, precision, recall, and F1 scores across the board. This technique also led to better generalization, as evidenced by the reduction in false negative rates (FNR) and false positive rates (FPR). 

Specifically, ResNet and MobileNet showed significant gains in accuracy and F1 scores, indicating enhanced ability to correctly classify both positive and negative instances. EfficientNet, which already performed well without augmentation, benefited from improved specificity and lower FPR, suggesting a more reliable performance in distinguishing true negatives. 


## Before Augmentation

| Model                         | Accuracy | Precision | Recall  | F1 Score | Specificity | FPR    | FNR    | PPV      | NPV      | MCC     | Youden's J | Markedness |
|--------------------------------|----------|-----------|---------|----------|-------------|--------|--------|----------|----------|---------|------------|------------|
| ResNet_No_Augmented           | 0.8375   | 0.858434  | 0.8375  | 0.835092 | 0.958333    | 0.041667 | 0.283333 | 0.945055 | 0.771812 | 0.695619 | 0.795833   | 0.716867   |
| EfficientNet_No_Augmented     | 0.8875   | 0.887742  | 0.8875  | 0.887482 | 0.875000    | 0.125000 | 0.100000 | 0.878049 | 0.897436 | 0.775242 | 0.762500   | 0.775485   |
| MobileNet_No_Augmentation     | 0.8125   | 0.822373  | 0.8125  | 0.811053 | 0.900000    | 0.100000 | 0.275000 | 0.878788 | 0.765957 | 0.634796 | 0.712500   | 0.644745   |


<div align="center">
  <img src="./figures/confusion_matrix_1.jpg" alt="Confusion Matrix without Data Augmentation">
</div>

## After Augmentation

| Model                   | Accuracy  | Precision | Recall   | F1 Score  | Specificity | FPR     | FNR     | PPV      | NPV      | MCC      | Youden's J | Markedness |
|-------------------------|-----------|-----------|----------|-----------|-------------|---------|---------|----------|----------|----------|------------|------------|
| ResNet_Augmented       | 0.895833  | 0.896081  | 0.895833 | 0.895817  | 0.883333    | 0.116667 | 0.091667 | 0.886179 | 0.905983 | 0.791914 | 0.779167   | 0.792162   |
| EfficientNet_Augmented | 0.895833  | 0.897185  | 0.895833 | 0.895745  | 0.925000    | 0.075000 | 0.133333 | 0.920354 | 0.874016 | 0.793017 | 0.820833   | 0.794370   |
| MobileNet_Augmented    | 0.858333  | 0.858433  | 0.858333 | 0.858323  | 0.850000    | 0.150000 | 0.133333 | 0.852459 | 0.864407 | 0.716766 | 0.708333   | 0.716866   |


<div align="center">
  <img src="./figures/confusion_matrix_2.jpg" alt="Confusion Matrix after Data Augmentation">
</div>


# Focusing on ResNet

The improvement observed in ResNet through data augmentation clearly indicates that the model can achieve even better results with further modifications (additional images). Data augmentation significantly enhanced ResNet's accuracy and F1 score, showcasing its ability to leverage the increased data variability for better learning and generalization. This suggests that by continuing to augment the dataset with more diverse and representative images, ResNet's performance can be further optimized.

Therefore, another test was conducted on the ResNet model by making minimal modifications to the learning rate and the number of epochs. Below are the latest results for the modified ResNet model.


| Model         | Accuracy  | Precision | Recall   | F1 Score  | Specificity | FPR     | FNR     | PPV      | NPV      | MCC      | Youden's J | Markedness |
|--------------|-----------|-----------|----------|-----------|-------------|---------|---------|----------|----------|----------|------------|------------|
| ResNet_Final | 0.920833  | 0.921097  | 0.920833 | 0.920821  | 0.908333    | 0.091667 | 0.066667 | 0.910569 | 0.931624 | 0.841930 | 0.829167   | 0.842193   |


<div align="center">
  <img src="./figures/resnet_confusion_matrix.jpg" alt="Resnet Confusion Matrix">
</div>

The metrics highlight a well-balanced and high-performing model. The accuracy, precision, recall, and F1 score are all consistently above 92%, showcasing ResNet's ability to correctly identify both positive and negative instances. The confusion matrix provides a visual representation of ResNet's performance. Out of 120 "notsmoking" images, ResNet correctly identified 111 and misclassified 9. For the "smoking" images, it correctly identified 109 out of 120, with 11 misclassifications.


# Discussion

The results suggest that ResNet's performance (in addition to EfficientNet) can continue to improve with further data images and fine-tuning. By adding more diverse and representative images, ResNet can enhance its ability to generalize, reducing the rates of misclassification even further.

In general, data augmentation had a positive impact on the performance of the machine learning models. By increasing the variety and amount of training data, augmentation improved the evaluation metrics of most models. It is evident that by augmenting the dataset, we were able to provide the models with more diverse examples, which helped them generalize better to unseen data. This led to improved accuracy, precision, recall, and other evaluation metrics. 













