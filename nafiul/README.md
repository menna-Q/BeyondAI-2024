![BeyondAI Banner for Research Projects](../BeyondAI_Banner_Research_Projects_2024.png)

### Description of Project  

**Motivation for Research Question**  
Diabetic Retinopathy (DR) is a leading cause of blindness worldwide, with over 1 million cases of blindness and 3.28 million severe vision impairments. Early detection is critical for mitigating its impact, especially in low-resource settings where access to healthcare may be limited. This project aims to address this challenge by evaluating machine learning models for effective DR detection, focusing on their applicability in clinical practices and constrained environments.

**Research Question**  
Which machine learning model, Convolutional Neural Networks (CNNs) or Vision Transformers (ViTs), performs better for detecting and classifying stages of Diabetic Retinopathy, particularly in low-resource settings?  

**Method and Implementation**  
1. **Dataset**: The Kaggle DR Dataset 2019 was utilized, comprising 224×224 retinal images categorized into five diagnostic stages of DR.  
2. **Image Augmentation**: Techniques such as random rotations, shifts, shearing, zooming, and horizontal flipping were applied to increase dataset diversity.  
3. **Model Architectures**:  
   - **CNN**: Extracts local features through convolutional layers, refines them via multiple layers, and categorizes images using fully connected layers.  
   - **ViT**: Divides images into patches, encodes spatial relationships with positional embeddings, and classifies images using self-attention mechanisms in Transformer blocks.  
4. **Evaluation Metrics**: Training and validation accuracy, F1-score, sensitivity, specificity, and confusion matrices were analyzed to compare model performances.  

**Results**  
- **Binary Classification (DR vs. No DR)**: CNNs exhibited higher training accuracy (91-95%) but significant drops in validation accuracy with smaller datasets, suggesting overfitting. ViTs maintained stable validation accuracy (~63-94%), indicating better generalization.  
- **Multiclass Classification (DR Stages)**: Both models displayed similar trends for F1-scores, sensitivity, and specificity. CNNs had higher training accuracy but plateaued validation accuracy, while ViTs showed consistent performance (~73%) for both metrics.  

**Conclusions**  
ViTs demonstrate superior generalization and robustness, making them more suitable for diverse, unseen cases in low-resource settings. However, CNNs excel in extracting local features and identifying specific DR stages, which can be advantageous in well-augmented datasets. Both models possess unique strengths, suggesting that model selection should be based on application requirements, dataset characteristics, and resource availability. Future work could explore hybrid approaches leveraging both models' complementary capabilities.  
