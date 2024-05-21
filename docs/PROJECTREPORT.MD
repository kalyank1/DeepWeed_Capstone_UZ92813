## WEEDSENSE - EMPOWERING PRECISION AGRICULTURE WITH COMPUTER VISION

A CAPSTONE PROJECT BY Kalyan Karre(UZ92813)
UNDER THE GUIDANCE OF  Dr. Muhammad Ali Yosuf
(Course Code: Data 606)

MASTER OF PROFESSIONAL STUDIES IN DATA SCIENCE
UNIVERSITY OF MARYLAND BALTIMORE COUNTY

## 1.	INTRODUCTION

Early detection and monitoring of weeds play a critical role in effective agricultural management. With the deployment of deep weed detectors in the field, farmers gain the ability to monitor weed populations in real-time, allowing them to identify emerging weed infestations before they escalate into widespread problems. This proactive approach empowers farmers to take timely intervention measures, thereby preventing potential crop yield losses. By leveraging computer vision technology, these detectors can swiftly and accurately identify weeds amidst crop plants, enabling farmers to respond promptly to localized weed outbreaks and apply targeted management strategies.
Moreover, the adoption of image-based weed detection systems offers significant potential for lowering production costs in agriculture. Traditional weed management methods, such as manual labor or blanket herbicide application, are often associated with high costs and inefficiencies. However, by implementing precision weed management techniques facilitated by deep weed detectors, farmers can minimize the need for excessive herbicide use or labor-intensive weed control measures. By accurately identifying and targeting specific weed-infested areas within fields, farmers can optimize resource allocation and reduce wastage, leading to cost savings and improved overall efficiency in crop production. This shift towards precision weed management not only reduces operational expenses but also contributes to environmentally sustainable farming practices by minimizing chemical inputs and mitigating the risk of herbicide resistance in weed populations.
The main aim of this study is to build an image classifier capable of detecting the weed species when weed plant is provided as an input.

## 2.	DATASET

I have used 2 datasets for this study. They are Deep Weeds and CottonWeedID15 datasets. This has been done to increase the richness of the data.
The Deep Weeds dataset consists of 17,509 images capturing 8 different weed species native to Australia in situ with neighboring flora. The selected weed species are local to pastoral grasslands across the state of Queensland. The images were collected from weed infestations at the following sites across Queensland: 
•	Black River
•	Charters Towers
•	Cluden
•	Douglas
•	Hervey Range
•	Kelso
•	McKinlay
•	Paluma.

The size of the Deep Weeds dataset is 494 MB.
The CottonWeedID15 dataset comprises 5187 RGB images capturing 15 common weed species typically found in cotton fields across the southern United States. 
•	Carpetweeds
•	Crabgrass
•	Eclipta
•	Goosegrass
•	Morningglory
•	Nutsedge
•	PalmerAmaranth
•	Prickly Sida
•	Purslane
•	Ragweed
•	Sicklepod
•	SpottedSpurge
•	SpurredAnoda
•	Swinecress
•	Waterhemp

The size of this CottonWeedID15 dataset is 11.35 GB. Since the dataset is huge, I have considered 50% of its size by taking 50% of images from each of the 15 class labels. Therefore, the total dataset size would be 494 MB (or 0.494 GB) + 5.675 GB (50% of 11.35 GB) = 6.169 GB.

## 3.	LITERATURE REVIEW 

Alex Olsen et al [2019] [1] studied multi class weed species image dataset using deep learning. This study addresses the gap in research on robotic weed control in rangeland environments by introducing the DeepWeeds dataset, the first large, publicly available multiclass image dataset of weed species from Australian rangelands. With 17,509 labeled images covering eight nationally significant weed species across eight locations in northern Australia, this dataset enables the development of robust classification methods essential for effective robotic weed control. The study employs benchmark deep learning models, namely Inception-v3 and ResNet-50, to establish a baseline for classification performance on the dataset. Results indicate high accuracy rates, with Inception-v3 achieving an average classification accuracy of 95.1% and ResNet-50 achieving 95.7%. Additionally, real-time performance of the ResNet-50 architecture is demonstrated, showcasing an average inference time of 53.4 milliseconds per image. These promising findings suggest a strong potential for the future implementation of robotic weed control methods in Australian rangelands, addressing critical challenges in weed management for stock farmers.

A S M Mahmudul Hasan et al [2021] [2] studied weed recognition using deep learning techniques on class imbalanced imagery. This paper investigates the efficacy of five state-of-the-art deep neural networks—VGG16, ResNet-50, Inception-V3, Inception-ResNet-v2, and MobileNetV2—for recognizing weeds from images, crucial for developing automatic weed management systems in agriculture. By employing various experimental settings and dataset combinations, including a large weed-crop dataset created by amalgamating smaller datasets and mitigating class imbalance through data augmentation, the study evaluates model performance. Transfer learning techniques, utilizing pre-trained weights for feature extraction, and fine-tuning on crop and weed datasets, are explored. Results indicate that VGG16 outperforms others on small-scale datasets, while ResNet-50 excels on larger combined datasets. The study underscores the significance of data augmentation and fine-tuning in enhancing deep learning model performance for crop and weed image classification.

## 4.	EXPLORATORY DATA ANALYSIS

Firstly, the distribution of the class labels from the combined dataset [Deep Weeds + 50% of CottonWeedID15] has been studied by plotting the bar chart as below.
It can be observed from the Fig 1, that the class label - Negative has the highest number of images while SpurredAnoda has the least number of images. Therefore, there is a huge imbalance in the dataset.
Fig 1 – Bar Chart showing distribution of class labels 


The following are the few sample images from the dataset to give an understanding about the images in the dataset.



Fig 2 – Sample images from the dataset

Further, the heights and widths of the images are studied by plotting the histogram of all image heights and widths as below.

 

Fig 3 – Histogram of Image heights and widths
It can be observed from the above plot that most of the image heights and widths lie in the range of 200-500.
The data has been augmented using ImageDataGenerator library in TensorFlow. Data augmentation is a pivotal technique in the realm of deep learning, particularly for computer vision tasks where training data is scarce. It operates by introducing variations into the training data, such as rotations, translations, flips, and zooms, thus augmenting the dataset's diversity. By doing so, it equips the model with a more comprehensive understanding of object orientations and scales, encouraging increased robustness.
Moreover, data augmentation plays a crucial role in enhancing the model's ability to generalize beyond the training set. By exposing the model to slightly altered versions of the training images, it mitigates the risk of overfitting, where the model excessively tailors itself to the training data. Instead, this technique encourages the model to extract more generalized features, leading to superior performance on unseen examples.
The following data augmentation operations are performed for the given train data.
•	Adjusting the brightness of the images within the specified range [0.5, 1.5].
•	Flipping images horizontally (left to right).

Now, let us see how these augmented images look like for a sample image. Here 12 augmented images are generated for one sample image.
 
 
 
Fig 4 - Augmented images for a sample image


## 5.	MODELLING 

To solve the defined problem, I have leveraged the transfer learning. Here, in this study VGG16 and InceptionV3 pretrained architectures have been considered.
The following approach has been followed for transfer learning.
•	Utilizing Pre-trained Model - Incorporating a pre-trained VGG16 and InceptionV3 models without the top layers as the foundation for weed species classification.
•	Customizing Architecture - Tailoring the model by adding additional fully connected layers to adapt it for precise weed species identification.
•	Fine-tuning for Optimization - Fine-tuning parameters such as learning rate and optimizer to enhance the model's accuracy and efficiency in classifying weed species.
•	Effective Evaluation Metrics - Assessing model performance using metrics like categorical cross-entropy loss and area under the ROC curve (AUC) to ensure accurate weed species classification.
•	Efficiency in Training - Leveraging transfer learning expedites the training process by utilizing pre-trained weights, leading to faster convergence and efficient resource utilization.


 
 
Fig 5 – VGG16 Architecture

  
Fig 6 – InceptionV3 Architecture

## 6.	RESULTS

The above models are trained using Adam as an optimizer with initial learning rate of 0.0001 using AUC as the performance metric. Here, AUC is used as the metric since we have severe imbalanced dataset.
Batch size of 16 has been considered. 

The following are the result plots obtained for VGG16.
•	At the convergence, @24 epochs
•	Initial Learning Rate – 0.0001
•	Optimizer – Adam 
•	Total params - 15799192 (60.27 MB)
•	Trainable params - 15799192 (60.27 MB)
•	Non-trainable params - 0 (0.00 Byte)

 
Fig 7 – Plots obtained using VGG16
The following are the result plots obtained for InceptionV3.

•	At the convergence, @16 epochs
•	Initial Learning Rate – 0.0001
•	Optimizer – Adam 
•	Total params: 24460152 (93.31 MB)
•	Trainable params: 24425720 (93.18 MB)
•	Non-trainable params: 34432 (134.50 KB)
 
Fig 8 – Plots obtained using InceptionV3

Model 	Train Loss	Test Loss	Train AUC	Test AUC
VGG16	1.0369	0.896	0.9729	0.9803
Inception V3	0.5444	0.5382	0.992	0.9918

From the above results, it is observed that AUC Score for test data obtained using InceptionV3 model is better than other model considered. Therefore, InceptionV3 model can be used for predictions.

Now, let us get some predictions using the Inception V3 since it has performed better.
These predictions are performed on the sample test data points.

 
Fig 9 – Predictions on sample test data


## 7.	CONCLUSION

Incorporating InceptionV3 into image classification systems offers a multitude of benefits. Its state-of-the-art architecture, high accuracy, and efficient feature extraction capabilities make it a top choice for developers seeking to build robust and accurate image classification models. By utilizing transfer learning with InceptionV3, developers can adapt the network to various classification tasks with minimal data requirements, saving time and computational resources. Overall, integrating InceptionV3 enhances the effectiveness and efficiency of image classification systems, providing businesses and researchers with powerful tools for solving complex classification challenges.


## 8.	REFERENCES

[1] Olsen, A., Konovalov, D. A., Philippa, B., Ridd, P., Wood, J. C., Johns, J., ... & White, R. D. (2019). DeepWeeds: A multiclass weed species image dataset for deep learning. Scientific reports, 9(1), 2058.
Link - https://arxiv.org/abs/1810.05726

[2] Hasan, A. M., Sohel, F., Diepeveen, D., Laga, H., & Jones, M. G. (2022). Weed recognition using deep learning techniques on class-imbalanced imagery. Crop and Pasture Science. 
Link - https://www.publish.csiro.au/cp/pdf/CP21626


