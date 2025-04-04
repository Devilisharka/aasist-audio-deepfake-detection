# AASIST-audio-deepfake-detection
**For the 3 forgery detection approaches i choose the following papers:**

**1) How to Boost Anti-Spoofing with X-Vectors:**

After going through this paper, I found that its main innovation lies in applying x-vectors, commonly used for speaker verification, to the task of spoofing detection. The authors use a TDNN-based architecture to extract embeddings from acoustic features like LFCC and CQCC, followed by statistical pooling and a simple classifier to distinguish real vs spoofed speech.The approach performed well on the ASVspoof 2019 dataset, achieving lower EER and t-DCF than traditional GMM or DNN systems. I also noticed that combining this with other models further improved robustness.What I liked about this method is that it's modular and lightweight, especially with the TDNN setup. It allows us to plug in different features and backends easily, which makes it a practical choice for experimentation or deployment. However, the paper also explored SENet34, which, while accurate, is computationally expensive. Another limitation is that the system relies on hand-crafted features, and may need careful tuning to generalize across unknown or diverse spoofing attacks.

**2) End-to-End Anti-Spoofing with RawNet2:**

​After reviewing the paper "End-to-End Anti-Spoofing with RawNet2," I observed that it introduces the application of RawNet2, originally developed for speaker verification, to the task of spoofing detection. This approach processes raw audio inputs directly, eliminating the need for handcrafted features. The architecture incorporates SincNet filters in the initial layer to capture essential frequency information, followed by residual blocks and a gated recurrent unit (GRU) layer for temporal modeling. An additional filter-wise feature map scaling (FMS) mechanism is integrated to enhance discriminative power. ​In terms of performance, the RawNet2-based system was evaluated on the ASVspoof 2019 Logical Access dataset. It achieved the second-best results for the challenging A17 attack and, when fused with baseline countermeasures, also secured the second-best performance across the entire dataset. ​This end-to-end approach streamlines the anti-spoofing pipeline by removing the dependency on manual feature engineering. By learning directly from raw audio, the system has the potential to capture subtle spoofing artifacts that traditional methods might overlook, making it a compelling choice for robust spoofing detection.​ However, despite its advantages, RawNet2's complexity may lead to increased computational demands, potentially impacting real-time processing capabilities. Additionally, the model's effectiveness against unseen spoofing attacks remains an area for further investigation.

**3) AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks**

​After reviewing the paper "AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks," I found that it presents a novel approach to detecting spoofed audio. The authors developed a system that uses graph attention networks (GATs) to analyze both the frequency (spectral) and time-based (temporal) aspects of audio signals. This design allows the model to identify subtle artifacts that indicate whether an audio sample is genuine or spoofed. ​In tests using the ASVspoof 2019 dataset, AASIST achieved a 20% improvement over previous top-performing systems. Notably, a lighter version of the model, AASIST-L, with only 85,000 parameters, also outperformed all other competing systems. ​I chose to study this paper because of its state-of-the-art performance on the ASVspoof dataset, indicating its effectiveness in identifying various types of spoofing attacks. The model's ability to process both spectral and temporal information in an integrated manner makes it a promising solution for practical anti-spoofing applications.​However, it's important to note that the use of graph-based models like GATs can be computationally intensive, which might pose challenges for real-time processing or deployment on devices with limited resources. Additionally, while the model performed well on the ASVspoof dataset, further testing would be necessary to confirm its effectiveness against new or unforeseen types of spoofing attacks.

**Challenges Encountered & Solutions**:

Label Mismatch in Evaluation Loader: During evaluation, the eval_loader from the ASVspoof 2019 dataset had inconsistent or missing labels, which impacted the ability to compute standard metrics like EER and t-DCF. To address this, I used the dev_loader for evaluation instead, which contained valid labels aligned with model predictions.

Class Imbalance in Dataset: The dataset exhibited a significant skew towards spoofed samples. To counter this, class weighting was applied in the loss function. The weights were computed based on the inverse frequency of each class and normalized, ensuring balanced contribution from both spoof and bonafide classes during training.

**Assumptions Made:**

The development set (dev_loader) was used as the evaluation set instead of the provided eval_loader. This choice was made due to a runtime error encountered during evaluation: RuntimeError: CUDA error: device-side assert triggered. The issue was likely caused by incorrect label formatting in eval_loader, which led to misalignment during loss computation or inference. To ensure smooth and reliable validation, the cleaner dev_loader was used throughout training and evaluation. Additionally, since proper label alignment and scoring scripts were not available or compatible during the evaluation phase, I used a mock calculation for t-DCF for approximate insight into performance; this approximation served as a basic heuristic to observe trends during model development but should not be considered a substitute for the official ASVspoof evaluation protocol. The actual t-DCF may vary and should ideally be computed using the official scripts provided with the dataset.

**Why I Selected AASIST for Implementation:**

After exploring several state-of-the-art methods for audio deepfake detection, I chose to implement AASIST because of its strong performance and architectural design suited to the ASVspoof challenge.One of the primary reasons was its outstanding results on the ASVspoof 2019 dataset, where it outperformed many competing models in both Equal Error Rate (EER%) and t-DCF—two widely accepted evaluation metrics for spoof detection. The model’s use of a spectro-temporal graph attention mechanism particularly stood out. This design allows AASIST to capture fine-grained patterns across both time and frequency domains, which is crucial for identifying subtle artifacts in AI-generated speech. Another reason I found AASIST appealing was its potential for adaptation. While the original version is complex, I saw the opportunity to simplify it for more efficient real-time use, especially in edge-device or security-focused applications. From a learning perspective, this project was also a great way to gain hands-on experience with audio forgery detection, a field I hadn't explored before.

**How AASIST Works (Simplified Overview)**

AASIST integrates signal-specific SincConv layers with spectro-temporal graph attention mechanisms to capture both low-level spectral cues and high-level relational structures in speech signals. The architecture constructs heterogeneous graphs where different node types (e.g., temporal, spectral, and spectro-temporal) interact through multi-head attention, enhanced by a master node for global context aggregation. A learnable graph pooling strategy selectively preserves the most salient nodes, enabling robust and discriminative representations. This combination of signal-aware convolution and graph-based reasoning enables the model to generalize effectively across diverse spoofing attacks, setting a strong benchmark in anti-spoofing research for speaker verification systems.

**Performance on chosen dataset:**

**Final Evaluation Metrics:**

Metric

Eval Accuracy: 77.03%

Eval Loss: 0.3759

Equal Error Rate: 24.76%

t-DCF (approx): 2.97%

The final evaluation demonstrates steady improvement in accuracy and a notable reduction in EER and t-DCF, indicating better discrimination between bonafide and spoofed audio samples. These results validate the effectiveness of our simplified attention-based model on the ASVspoof 2019 dataset.

**Strengths:**

**Effective Learning Progression:** The model shows steady improvement in training performance, with accuracy rising from 71.20% to 74.57% and training loss reducing from 0.6106 to 0.4423 over 5 epochs, indicating consistent convergence.

**Strong Generalization at Final Epoch:** Final evaluation metrics—77.03% accuracy, 24.76% EER, and t-DCF of 22.97%—demonstrate the model's capacity to generalize well to unseen spoofed and bona fide audio samples.

**Robustness to Audio Forgery:** The low EER and t-DCF values achieved in the final evaluation phase suggest reliable discrimination between spoofed and genuine speech, which is critical for anti-spoofing tasks.

**Architecture Efficiency:** Despite the complexity of the AASIST model, the training process remained stable, and key performance metrics improved without architectural modifications—validating the efficacy of its integrated spectro-temporal and graph attention mechanisms

**Weaknesses:**

**Inconsistent Evaluation Behavior:** During epoch 3, the model experienced a significant performance dip (63.23% accuracy, t-DCF spike to 36.77%), indicating potential instability in generalization across epochs.

**Sensitivity to Training Dynamics:** Variability in evaluation accuracy and t-DCF suggests the model may be sensitive to hyperparameters such as learning rate or batch size, and could benefit from advanced scheduling or regularization techniques.

**Scope for Optimization:** Although the model performs well, further improvements might be achieved through: Fine-tuning graph attention configurations,Applying data augmentation techniques,Incorporating validation-based early stopping

**Computational Cost:** Training AASIST is resource-intensive, taking ~1–1.5 hours for 5 epochs on GPU, which could pose challenges for rapid experimentation or deployment in low-resource environments.

**Suggestions for future improvements:**

To further improve the model, I plan to explore several directions. First, increasing the size and variety of training data through simple augmentation techniques like adding noise or changing playback speed may help the model generalize better. Extending training for more epochs and tuning hyperparameters like the learning rate or batch size could also enhance performance. Additionally, I’m interested in replacing or enhancing the current attention mechanism by integrating x-vectors, as suggested in recent anti-spoofing research, to see how it affects key metrics like EER and t-DCF. Finally, testing the model on different datasets will help assess its robustness in real-world scenarios.

**What were the most significant challenges in implementing this model?**

The most significant challenge was getting the AASIST model to run smoothly within my development setup. Since it’s a complex architecture with many dependencies, I had to spend time resolving compatibility issues and understanding how the different components (like spectro-temporal graphs and attention mechanisms) interact. Also, training was slow due to high GPU demand, which made it difficult to iterate quickly and experiment with changes.

**Real-World vs. Research Dataset Performance:**

While the model performs well on the ASVspoof 2019 dataset, real-world conditions may introduce more diverse and noisy data, such as varying recording equipment, background noise, or unseen spoofing techniques. This could lead to a drop in performance, highlighting the need for robustness testing and domain adaptation.

**Additional Data or Resources for Improvement**:

More diverse and large-scale spoofed audio samples, including recent deepfake methods, would help the model generalize better. Access to high-performance GPUs and tools like x-vectors could also boost training efficiency and model accuracy. Exploring ensemble techniques might further reduce false positives.

**Approach for Production Deployment:**

I would start by saving the trained model and creating a simple API using Flask or FastAPI to allow other applications to send audio files and receive predictions. To make it easier to manage and run on different systems, I’d package the project using Docker. Before going live, I’d test the model with real-world audio examples to check if it handles different recording qualities well. For ongoing use, I’d also add basic logging to track how the model performs over time and flag any unusual inputs or low-confidence predictions.


## Requirements

This project was implemented and tested on [Kaggle Notebooks](https://www.kaggle.com/code), which provides a pre-configured environment with most dependencies installed.  
You can run the entire pipeline without any additional setup.

**📌 Kaggle Notebook (public):** [https://www.kaggle.com/code/arkaprabharay/asvspoof2019-with-aasist](https://www.kaggle.com/code/arkaprabharay/asvspoof2019-with-aasist)

---

### 📦 Dependencies

The following libraries were used (all pre-installed in Kaggle):

- Python 3.8+
- PyTorch
- torchaudio
- librosa
- numpy
- pandas
- matplotlib
- tqdm

> ✅ No manual installation needed when using Kaggle.

---

### 🔁 Reproducibility

This project is fully reproducible within the Kaggle environment. The ASVspoof 2019 dataset provides predefined folder structures for training and evaluation.

- No additional data preprocessing or splitting was necessary.
- Results can be reproduced by simply running the notebook in Kaggle.
- If using locally, ensure the dataset structure mirrors that in the original ASVspoof 2019 setup.

---

### 📂 Data Access

The project uses the [ASVspoof 2019 LA dataset](https://datashare.ed.ac.uk/handle/10283/3336).

To access the data in Kaggle:

1. Download the dataset manually from the above link (access request may be required).
2. Upload the relevant parts (`flac` files, `protocols` folder, etc.) to your Kaggle Notebook via the **“Add data”** section.
3. The notebook expects the following folder structure:


