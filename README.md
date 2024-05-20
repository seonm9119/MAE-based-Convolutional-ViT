## MAE-based Hybrid Convolutional ViT for Self-Supervised Learning

### Motivation
The Necessity of Developing Lightweight Models

- In the field of image recognition and computer vision, deep learning models have significantly improved performance. However, these models have become larger in size and require more computational resources.
- Self-supervised learning is a method of training models without explicit labels. It involves using two networks, such as an encoder and decoder (e.g., autoencoder-based) or Siamese network. Compared to supervised learning, self-supervised learning requires more computational resources.

In recent years, deep learning models in the field of computer vision have seen significant improvements in performance, but along with that, the required resources have also increased. In particular, self-supervised models, which train without labels, typically use two networks, making them more resource-intensive compared to supervised models.
The table below illustrates the pre-training time and memory usage of some self-supervised learning models. Even when using the smallest backbone of Vision Transformers with 22 million parameters, training the model requires 312G of memory and takes 8 days. Recent state-of-the-art models, which typically employ around 100 million parameters, demand even more memory resources.

<p align="center">
  <img src="https://github.com/seonm9119/seonm9119.github.io/assets/125437452/f64c51ad-6aac-45ea-873c-15c5fe799ab9" alt="figure1" width="80%">
</p>
<p align="center">
 "Image BERT Pre-training with Online Tokenizer." International Conference on Learning Representations. 2022
</p>

---

### CvT: Introducing Convolutions to Vision Transformers

CvT (Convolutional Vision Transformer) is based on the convolutional operation, which is added to the existing Vision Transformer architecture to model local information and relationships between the entire image and patches. CvT is structured hierarchically, and each layer consists of convolutional token embeddings that perform nested operations. Unlike Vision Transformers composed of non-overlapping patches, the token map reconstructed in 2D in CvT reflects the local information of the image, leading to better performance.

<p align="center">
  <img src="https://github.com/seonm9119/seonm9119.github.io/assets/125437452/928b4079-c936-4647-a636-080fbafb4f0d" alt="figure2" width="80%">
</p>
<p align="center">
"CvT: Introducing convolutions to vision transformers." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
</p>

Looking at the evaluation table of CvT model performance, it demonstrates superior performance compared to Vision Transformers with fewer parameters and computational costs. Considering these performance and cost aspects, CvT is worth noting as a valuable backbone for self-supervised models.

<p align="center">
  <img src="https://github.com/seonm9119/seonm9119.github.io/assets/125437452/2f6aa462-18d9-4762-baeb-e33cd3db7256" alt="figure2" width="80%">
</p>
<p align="center">
"CvT: Introducing convolutions to vision transformers." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
</p>

---

### Image Inpainting vs Masked Image Modeling
- Image inpainting is a technique used to restore damaged or missing parts of an image.
- Masked Image Modeling is a task where certain regions of an image are masked, and the model predicts those regions. 
- MIM demonstrates excellent performance in various image processing tasks and can be improved by leveraging state-of-the-art models and datasets.

#### The key elements of inpainting: 
- Update Mask 
- Skip-Connections

<p align="center">
  <img src="https://github.com/seonm9119/seonm9119.github.io/assets/125437452/ffcab8ba-78f6-4b18-a545-1ba79b9424e7" alt="figure2" width="50%">
</p>

<p align="center">
  <img src="https://github.com/seonm9119/seonm9119.github.io/assets/125437452/73887b07-1ec0-4f0a-ba6b-ea71319873aa" alt="figure2" width="50%">
</p>


In recent years, Masked Image Modeling (MIM) has been utilized in various computer vision tasks and has shown excellent performance. MIM involves masking certain regions of an image and training the model to restore the masked regions. Similarly, image inpainting is a classical computer vision task that aims to restore missing parts of an image. In the case of image inpainting, to successfully reconstruct the image, the model needs to learn features of the image itself. Therefore, inpainting can be considered as a powerful tool, similar to self-supervised models, for learning image features. Thus, in this study, we focused on the differences rather than the similarities and explored how to apply the working principles and model architecture of inpainting to self-supervised models.

---

### Image Inpainting

The key elements of inpainting include the update mask and skip connections. The diagram illustrates the process of the update mask. The update mask is a binary mask that indicates which pixels of the image have been filled in during the inpainting process. By iteratively updating the mask and inpainting the remaining areas, the network progressively fills in the missing parts of the image.

---

### Model Architecture

<p align="center">
  <img src="https://github.com/seonm9119/seonm9119.github.io/assets/125437452/a1734326-f523-404a-bb1e-fb062d63817a" alt="figure2" width="50%">
</p>

The proposed model in this paper utilizes the CVT backbone and adopts the model architecture and working principles of inpainting. The overall structure of the model consists of an encoder and a decoder. Each layer of the encoder incorporates the update mask strategy, while in the decoder, the information from each layer of the encoder is concatenated with the decoder, resembling a structure similar to U-Net.

---


### Experiment

To evaluate the performance of the model, three experiments were conducted. Firstly, the influence of the update mask and skip connections on the model's performance was investigated. The experimental results confirmed that skip connections and the update mask strategy improve the model's performance. Secondly, it was empirically demonstrated that the proposed model exhibits stable learning on smaller images compared to vision transformer-based models. Based on this finding, a novel training strategy is proposed.

1. Exploration of Model Performance with Updated Masks and Skip Connections:
  - We investigate the performance of the model by utilizing updated masks and skip connections.
  - These technical elements play a crucial role in enhancing the accuracy and stability of the model.

2. Model Lightweighting and Resource Efficiency with CvT:
  - We utilize CvT to lighten the model and achieve resource savings.
  - This enables efficient learning while conserving computational resources.

3. Demonstrating Improved Stability in Learning from Smaller Images:
  - We demonstrate that our proposed model exhibits more stable learning from smaller images compared to existing ViT-based models.
  - Through this, we propose new learning strategies.


---


#### Experimenting the Impact of Update Masking and Skip Connections on Model Performance

The results of this experiment reveal two interesting facts.

1. Comparison based on the Application of Skip Connections (1st and 2nd rows vs 3rd and 4th rows) 
   - In fine-tuning, +0.71 to 1.08, In linear classification, +10.9 to 10.23.
2. Comparison based on the Application of Update Masking (3rd row vs 4th row)
   - In linear classification, +1.46.


<p align="center">
  <img src="https://github.com/seonm9119/seonm9119.github.io/assets/125437452/1b11618a-d94e-460c-9c70-c1f31b487de1" alt="figure2" width="50%">
</p>

We first conducted an experiment to investigate the impact of the update mask and skip connections on the model's performance. The experiment utilized the Tiny-ImageNet-200 dataset and the model was pre-trained for 500 epochs. The table presents the results for fine-tuning and fixed linear classification. Comparing the third and fourth rows, which correspond to models with skip connections, with the first and second rows, which lack skip connections, it is evident that the model with skip connections shows an improvement of over 10% in performance for fixed linear classification. Furthermore, comparing the results of the model with skip connections based on whether the update mask was applied or not, it can be observed that the model's performance improved by approximately 1% in linear classification. These results confirm that skip connections and the update mask strategy contribute to better learning of image representations.

---


#### Comparison of State-of-the-Art (SOTA) in Tiny-ImageNet-200

The model we proposed demonstrates outstanding performance with limited resources and epochs.

<p align="center">
  <img src="https://github.com/seonm9119/seonm9119.github.io/assets/125437452/6c3304d1-6ebd-492a-b364-cda278be1fd3" alt="figure2" width="50%">
</p>

The second experiment presents the results of comparing the proposed model with a state-of-the-art model on Tiny-ImageNet-200. The proposed model demonstrated superior performance compared to the state-of-the-art model. However, these results were attributed to the influence of patch size caused by the small image size. Therefore, additional experiments were conducted to investigate the effects of patch size and masking size.

#### Comparison based on Patch Size and Masking Size.

- It can be observed that as the patch size and masking size decrease, the model's performance improves. However, smaller patch sizes require more computational operations.
- In the case of ViT or Swin, since they mask a single token without overlap, the model's performance is influenced by the patch size and the masking region.


<p align="center">
  <img src="https://github.com/seonm9119/seonm9119.github.io/assets/125437452/add0b41e-1eb9-4c0e-ac97-26d3c6fbcd9b" alt="figure2" width="50%">
</p>

The table above presents the results based on different patch sizes and masking sizes. It can be observed that as the patch size and masking size decrease, the model's performance improves. However, the proposed model shows less sensitivity to patch size and masking size compared to other models. This interpretation can be attributed to the functioning of the CVT backbone used. In the case of the CVT backbone, overlapping 2D grid maps are used as tokens, which results in less sensitivity to the patch size and masking size compared to models that utilize different backbones.

---


#### WHY? The proposed model is less affected by patch size and masking size.

An interesting observation is that despite pretraining with only 1/4 of the images, similar results were obtained compared to training with the entire dataset.


<p align="center">
  <img src="https://github.com/seonm9119/seonm9119.github.io/assets/125437452/9c7f75d1-5dee-465f-ab13-3ab84e41f3fd" alt="figure2" width="50%">
</p>

Based on the previous experiments, it was empirically confirmed that the proposed model is less sensitive to performance degradation due to patch size and masking size compared to models that use ViT or Swin as backbones. Building upon these results, a simple experiment was designed using only 10% of the data from each class in Imagenet-1k. The first row in the table represents the results of pre-training on 224-sized images followed by fine-tuning on 224-sized images. The second row shows the results of randomly cropping images to a size of 64, pre-training the model, and then fine-tuning on 224-sized images. Interestingly, even though only 1/4 of the entire image was used for pre-training, similar results to training on the full-size images were obtained. This approach was possible because the proposed model is less affected by patch size and masking size, allowing for training the model on smaller images.

<p align="center">
  <img src="https://github.com/seonm9119/seonm9119.github.io/assets/125437452/30afe376-ff13-4e36-9e56-f1222565adbe" alt="figure2" width="50%">
</p>
<p align="center">
  <img src="https://github.com/seonm9119/seonm9119.github.io/assets/125437452/e353b3af-fb9b-424f-bf72-9137a05f0185" alt="figure2" width="50%">
</p>

The table above presents the results of comparing the proposed model with state-of-the-art models on ImageNet-1k. The first table compares the proposed model with models adopting the autoencoder approach, while the second table shows the performance of models with similar parameter counts. The proposed model demonstrates the results of pre-training on randomly cropped 64-sized images followed by fine-tuning. Although the proposed model exhibits relatively lower performance on ImageNet-1k compared to other models, it consumes significantly less memory and pre-training time. These results suggest that the proposed approach can be utilized as a new strategy to efficiently utilize resources during the pre-training phase of self-supervised learning models. It also highlights the potential to train efficient models in resource-constrained scenarios through the development of lightweight models and the application of new training strategies.

---
