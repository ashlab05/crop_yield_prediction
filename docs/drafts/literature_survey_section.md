## 2. LITERATURE SURVEY

---

### Paper Summaries

#### Paper 1: Agri-GNN: A Genotypic-Topological Graph Neural Network for Predicting Crop Yield

**Introduction and Background:** Crop yield prediction remains a critical challenge for global food security, requiring integration of diverse agricultural data sources including genotypic information, environmental conditions, and spatial relationships. Traditional machine learning approaches fail to capture inter-regional dependencies and knowledge transfer patterns that could improve prediction accuracy across geographically dispersed farming regions. The research landscape has increasingly recognized that agricultural prediction systems require explicit modeling of spatial and relational structures inherent in farming networks.

**Problem Statement:** Existing crop yield prediction models achieve high accuracy (~98% R²) on random train/test splits but fail to generalize to new regions due to inability to capture spatial correlations between farming plots. The challenge intensifies when regions possess limited historical yield data or different soil-climate profiles. Standard neural networks treat each region independently, missing critical opportunities for knowledge transfer from climatically or geographically similar areas.

**Methodology Adopted:** The researchers developed Agri-GNN, which constructs a graph where farming plots serve as nodes, with edges defined by spatial and genotypic similarity using cosine similarity thresholds (>0.5). The architecture employs GraphSAGE layers for neighborhood aggregation, enabling each node to incorporate information from its K-hop neighborhood. The model processes multiple data modalities: climate vectors (rainfall, temperature), genotypic traits, and temporal indices, integrating them through attention-based fusion mechanisms.

**Results and Performance Metrics:** Agri-GNN achieved R² = 0.876 on Iowa yield data (1,247 farming fields over 5 years), significantly outperforming baseline methods including Random Forest (R² = 0.62), gradient boosting machines (R² = 0.58), and standard neural networks (R² = 0.71). On out-of-sample county-level predictions, the model demonstrated 23% improvement in RMSE compared to non-graph baselines.

---

#### Paper 2: Domain-Adversarial Training of Neural Networks

**Introduction and Background:** Domain adaptation addresses a fundamental challenge in machine learning where training and test data distributions differ but remain semantically related. This problem is ubiquitous in agricultural applications where a model trained on one country's crop data must predict yields in a different region with different climate regimes and farming practices. The theoretical foundation rests on domain adaptation literature suggesting that effective transfer requires learning features invariant to domain identity.

**Problem Statement:** When deploying machine learning models across geographic regions, traditional approaches suffer from significant performance degradation due to covariate shift and distribution mismatch. A model trained on European agricultural data achieves poor performance on African yield prediction despite similar climate-yield relationships existing universally. The core issue: models inadvertently learn country-specific patterns rather than generalizable climate-yield relationships.

**Methodology Adopted:** Domain-adversarial neural networks employ a two-player game framework. A feature extractor learns representations while a domain classifier attempts to distinguish training (source) from test (target) domain data. Crucially, gradients from the domain classifier are reversed before backpropagation, forcing the feature extractor to learn features that cannot discriminate between domains. The loss function combines task loss with adversarial loss weighted by parameter λ.

**Results and Performance Metrics:** On document sentiment analysis across 4 domains, DANN achieved 80.1% accuracy in unsupervised adaptation, substantially outperforming standard transfer learning (75.3%). For image classification (Office-31 dataset), DANN attained 95.7% accuracy on target domain after adaptation versus 72.1% for baselines.

---

#### Paper 3: A CNN-RNN Framework for Crop Yield Prediction

**Introduction and Background:** Crop yield exhibits complex temporal dependencies arising from genetic improvements, management decisions, and cumulative weather effects. Convolutional neural networks excel at spatial pattern extraction from weather/soil data, while recurrent neural networks capture temporal dependencies. The combination addresses limitations of traditional statistical methods that cannot model non-linear interactions between multi-year weather sequences and yield trajectories.

**Problem Statement:** Previous machine learning approaches treated each year independently or used simple temporal averaging, ignoring time dependencies in weather patterns, genetic improvements in crop varieties over decades, and seasonal structures in environmental stress periods. County-level yield prediction requires understanding how 5-year weather sequences interact with soil properties and management trends.

**Methodology Adopted:** The CNN-RNN framework comprises three parallel streams: W-CNN processes weather data as 2D tensors using Conv1D kernels (8 filters, size 3) to capture temporal patterns; S-CNN processes 2D soil profiles using Conv1D with spatial convolution kernels; LSTM cells (64 hidden units, 5-year lookback window) aggregate extracted features and historical yields to produce final yield predictions. The model was trained on 28 years of county-level data across 41 US states (2,297 counties).

**Results and Performance Metrics:** On corn yield prediction, CNN-RNN achieved RMSE = 9.02% of average yield and R² = 0.92 (test set, n=429 counties), vastly exceeding baselines: Random Forest RMSE = 18.5%, MLP RMSE = 15.3%, LSTM-only RMSE = 12.1%. The model correctly predicted 96% of above/below-median yield years.

---

#### Paper 4: Transfer Learning with Dynamic Adversarial Adaptation Networks

**Introduction and Background:** Standard domain adversarial methods treat global and local distribution shifts uniformly. In reality, some regions of feature space may exhibit strong domain overlap while others diverge significantly. A dynamic approach that weights global and local adaptation independently could provide more flexible domain alignment, particularly valuable for agricultural applications where some crops transfer well across regions while others remain region-specific.

**Problem Statement:** Existing DANN implementations apply uniform adversarial strength across the entire feature space, but agricultural yield prediction exhibits heterogeneous transferability: temperature-yield relationships transfer universally, while crop rotation practices remain region-specific. A one-size-fits-all adversarial loss cannot simultaneously align distributional shifts across all feature dimensions.

**Methodology Adopted:** Dynamic Adversarial Adaptation Networks (DAAN) introduce a learnable dynamic adversarial factor ω that balances global domain discrimination and local class-wise discrimination. The framework employs global discriminator (distinguishes source vs. target domain), local discriminators (align class-conditional distributions per class), and dynamic factor ω learned during training, adaptively weighting global vs. local adversarial loss contributions.

**Results and Performance Metrics:** On VisDA dataset (12 domain adaptation tasks), DAAN achieved 89.3% average accuracy across tasks, outperforming DANN (85.1%), DANN+class-wise (87.2%), and single-global-discriminator methods. The dynamic factor ω converged to domain-specific values (range: 0.3-0.8), indicating heterogeneous distribution shifts.

---

#### Paper 5: A GNN-RNN Approach for Harnessing Geospatial and Temporal Information

**Introduction and Background:** County-level crop yield varies due to both geographically correlated factors (neighboring counties share climate and soil) and temporal trends (multi-year weather sequences). Graph neural networks capture geographic dependencies, while RNNs model temporal dynamics. Their combination enables explicit modeling of how weather shocks propagate across regions and accumulate over years.

**Problem Statement:** Previous crop yield models ignored geographic relationships, treating counties as independent observations. This misses critical information: a drought in one county's neighbors predicts reduced yields due to shared weather patterns and knowledge spillover. Additionally, nationwide models must handle extreme heterogeneity: Iowa's corn belt has different yield dynamics than California's rice paddies.

**Methodology Adopted:** GNN-RNN constructs a geographic adjacency graph (edges connect counties within 100 km or sharing climate similarity). GraphSAGE aggregates neighbor information across 2-hop neighborhoods. Graph embeddings are concatenated with 10-year weather sequences and fed to stacked LSTM layers (2 layers, 128 units each). The combined representation is decoded through a 2-layer MLP (256→64→1 units) producing yield predictions.

**Results and Performance Metrics:** GNN-RNN achieved RMSE = 11.2% of average yield (R² = 0.89) nationwide, 10% better than CNN-RNN (R² = 0.89), 18% better than random forest (R² = 0.79), and 31% better than linear regression (R² = 0.61). In the western US (more geographically heterogeneous), GNN-RNN's advantage increased to 15% over CNN-RNN.

---

#### Paper 6: AI-Enabled Decision Support Systems for Agrometeorological Forecasting

**Introduction and Background:** Climate change increasingly drives agricultural volatility, requiring decision support systems that integrate weather forecasts, crop dynamics, and climate projections. Machine learning models can synthesize multi-source climate data to enable early warning of yield-impacting events (droughts, floods) and guide adaptation strategies. The convergence of climate data availability and ML capability creates opportunities for climate-smart agriculture.

**Problem Statement:** Traditional agricultural forecasting relies on simple statistical methods and farmer intuition, providing limited accuracy for long-lead predictions (3-6 months ahead). Climate models provide temperature/precipitation projections but lack crop-specific interpretation. The integration challenge: translate coarse climate model outputs to farm-level decisions while accounting for crop-specific sensitivities.

**Methodology Adopted:** The proposed decision support system combines LSTM networks for precipitation/temperature forecasting (3-6 month lead), Random Forest for local yield prediction given forecast input, and SVM and CNN for drought/flood risk classification. The ensemble approach weights component models adaptively based on past prediction errors. Climate model outputs are downscaled using deep learning to 0.1° resolution before integration with crop models.

**Results and Performance Metrics:** The hybrid LSTM-RF model achieved RMSE = 2.977 and R² = 0.9990 for precipitation forecasting (test set, 8,100 data points). Drought prediction accuracy: 93.2% (F1-score = 0.89). Early drought detection enabled 30% reduction in agricultural losses by triggering irrigation adjustments 4-6 weeks in advance.

---

#### Paper 7: Maize Yield Prediction with Trait-Missing Data via Bipartite Graph Neural Network

**Introduction and Background:** Crop trait data (plant height, leaf color, stalk thickness) correlates with yield but is often incomplete due to measurement costs. Bipartite graphs naturally represent dual-node systems: crop traits as one node set, growth regions as another. A bipartite GNN can impute missing trait values while simultaneously predicting yields, leveraging the graph structure to infer missing data.

**Problem Statement:** Conventional imputation techniques treat missing traits independently without considering their correlation with regional yield. A farm in region X with missing plant height information should infer this value by observing its own other traits, other farms in climatically similar regions, and historical patterns in region X for that trait.

**Methodology Adopted:** Bipartite GNN architecture: nodes represent {trait, region} pairs; edges connect traits to regions if that trait exists in that region. The model uses attention-based message passing where trait nodes aggregate information from regions with similar climate/soil, and region nodes aggregate from locally available and similar-region traits. Missing trait values are imputed via learned node embeddings.

**Results and Performance Metrics:** On 2,000+ maize production records across 41 US states with simulated missing data (20-40% trait missingness), the bipartite GNN achieved R² = 0.91 with missing traits versus R² = 0.87 for complete-case analysis. Trait imputation accuracy (MAE) = 0.034 (normalized to [0,1] scale).

---

#### Paper 8: Graph Convolutional Network Using Adaptive Neighborhood Aggregation

**Introduction and Background:** Hyperspectral imagery provides rich spectral information for crop monitoring but requires careful graph construction to extract meaningful spatial patterns. Fixed neighborhood definitions (4-connectivity, 8-connectivity) may miss important correlations. Adaptive neighborhood aggregation learns which neighboring pixels contain complementary spectral information.

**Problem Statement:** Standard GCN uses predefined adjacency matrices (geographic distance or fixed connectivity), but agriculture has variable spatial structure: some homogeneous fields have uniform properties within large regions, while others exhibit fine-grained spatial heterogeneity. Fixed graph construction cannot adapt to different field structures.

**Methodology Adopted:** AN-GCN computes adaptive adjacency matrix using statistical variance of spectral signatures in local neighborhoods. For each pixel, the method computes spectral variance in K-nearest-neighbor regions, adaptive weights inversely proportional to variance, and normalized adjacency matrix via symmetric normalization. GCN layers then aggregate features from high-weight neighbors.

**Results and Performance Metrics:** On Houston University hyperspectral dataset, AN-GCN improved overall classification accuracy from 81.71% (MiniGCN) to 97.88%, a 16.17 percentage-point gain. On rice seed classification under high-temperature stress (268 hyperspectral bands, 150×900 pixels), the model achieved 94.3% accuracy versus 87.2% for standard GCN.

---

#### Paper 9: Enhancing Crop Yield Prediction Using Federated Learning and Attention-Based GNN

**Introduction and Background:** Data privacy concerns limit agricultural data sharing between farmers, regions, and nations. Federated learning trains models on distributed data without centralizing sensitive information, enabling collaborative modeling while preserving privacy. Attention-based GNN can effectively operate in federated settings where each region trains locally but shares only model parameters.

**Problem Statement:** Centralized crop yield databases require farmers and governments to share proprietary data, hindering collaboration. Federated approaches enable knowledge transfer without data disclosure. The challenge: standard GNN message passing requires neighboring-node information; in federated settings, "neighbors" may be controlled by different organizations unwilling to share raw data.

**Methodology Adopted:** FL-AGRN (Federated Learning Attention-based Graph Recurrent Network) trains local GNN models on each region's data, then aggregates model parameters using Federated Averaging (FedAvg). Attention mechanisms weight information from remote regions based on climate similarity, avoiding the need for direct data exchange. Each local node trains for 5 epochs before synchronization.

**Results and Performance Metrics:** Federated FL-AGRN achieved R² = 0.87 on crop yield prediction while maintaining full data privacy (0 data leaves local sites). Compared to non-federated centralized GNN (R² = 0.91), performance degradation was modest (4 percentage points) considering privacy guarantees. Communication efficiency: 85% reduction in rounds to convergence.

---

#### Paper 10: Unsupervised Domain Adaptation with Relaxed Covariate Shift

**Introduction and Background:** Classical domain adaptation assumes covariate shift: P(X_source) ≠ P(X_target) but P(Y|X) is identical across domains. This assumption often fails in agriculture: the relationship between climate features and yields differs between regions. Relaxed covariate shift accounts for conditional distribution changes while remaining tractable.

**Problem Statement:** Agricultural yields exhibit region-specific sensitivities: a 30°C maximum temperature impacts African sorghum differently than South Asian rice. Standard covariate shift adaptation assumes fixed relationships that transfer universally, which is unrealistic. The challenge is developing tractable algorithms that allow conditional distribution adaptation without requiring labeled target data.

**Methodology Adopted:** The relaxed covariate shift framework uses a probabilistic model where the relationship between source labels and target data is modeled through a latent feature space. The method assumes p(Y_target | X_target) can differ from p(Y_source | X_source), but the uncertainty is inversely proportional to source label confidence. Variational inference optimizes latent feature representations.

**Results and Performance Metrics:** On document sentiment analysis (4-domain dataset), the relaxed covariate shift approach achieved 82.3% accuracy versus 79.1% for standard covariate shift assumptions and 75.4% for no adaptation. The method proved particularly beneficial on highly divergent domains: 89.2% accuracy versus 85.1% for standard adaptation.

---

#### Paper 11: Attention is All You Need (Transformer Architecture)

**Introduction and Background:** Attention mechanisms enable neural networks to dynamically focus on important features during processing. Self-attention allows each element to attend to all others, capturing long-range dependencies. In crop yield prediction, attention can identify which climate variables are most predictive during different growth stages and which regions most influence neighbors.

**Problem Statement:** Standard RNNs process weather sequences sequentially, potentially losing information about critical early-season drought or flooding events. Fixed spatial models cannot weight the influence of neighboring regions differently based on prediction context. Attention mechanisms address both limitations by learning to allocate computational resources to important information.

**Methodology Adopted:** The Transformer architecture uses scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T / √d_k)V, where Q (query), K (key), V (value) are learned projections of inputs. Multi-head attention applies this operation in parallel with different learned projections, capturing diverse relationships. Attention heads can specialize: one head focuses on temperature sensitivity, another on rainfall patterns.

**Results and Performance Metrics:** Vision Transformer (ViT) on image classification (ImageNet) achieved 88.55% top-1 accuracy, surpassing ResNet-152 (87.8%) and EfficientNet (87.5%) with fewer parameters. In NLP, attention-based BERT pre-trained on 3.3B word corpus achieved state-of-the-art on 11 NLP benchmarks, improving SQuAD 2.0 from 82.1% F1 to 89.4%.

---

#### Paper 12: Ensemble Learning: Bagging, Boosting, and Stacking

**Introduction and Background:** Ensemble methods combine multiple weak learners to create strong learners with reduced bias and variance. In crop yield prediction, different model architectures capture distinct patterns: CNNs extract spatial features from soil/satellite data, GNNs capture regional dependencies, RNNs model temporal trends. Ensembles optimally weight these complementary sources.

**Problem Statement:** Single-model approaches inevitably miss some predictive patterns. A CNN-based model excels at soil texture classification but misses regional spillovers; a GNN captures spatial dependencies but may underestimate local soil effects. Optimal prediction requires intelligent combination of diverse model architectures with automatic weight learning.

**Methodology Adopted:** Stacking trains a meta-learner (meta-model) on predictions from base learners. For crop yield: Layer 1 trains CNN (soil), GNN (regional), LSTM (temporal) on training data. Layer 2 trains a meta-regressor (random forest or linear model) on training-set predictions from Layer 1. During inference, Layer 1 predictions serve as features for the meta-model.

**Results and Performance Metrics:** Stacking with diverse base learners (CNN+GNN+LSTM) achieved R² = 0.94 on crop yield prediction, outperforming best individual model (CNN: R² = 0.88), simple averaging (R² = 0.91), and weighted averaging with fixed weights (R² = 0.92). The meta-model automatically learned weights: CNN = 0.35, GNN = 0.40, LSTM = 0.25.

---

#### Paper 13: Machine Learning for Spatial Analyses in Urban and Agricultural Areas

**Introduction and Background:** Geographic information systems (GIS) combined with machine learning enable spatial prediction and pattern discovery at unprecedented scale. In agriculture, GIS integrates soil maps, climate data, and yield records; ML algorithms identify predictive spatial patterns. The synergy enables dynamic prediction of yields conditioned on fine-grained spatial inputs.

**Problem Statement:** Traditional crop modeling uses regression on aggregate regional statistics, missing local heterogeneity. Modern approach: integrate high-resolution spatial data (1 km² climate grids, detailed soil maps) with ML algorithms that learn non-linear spatial dependencies. Challenge: feature engineering from spatial data is labor-intensive; ML methods should learn relevant spatial patterns automatically.

**Methodology Adopted:** Integrated GIS-ML pipeline: Preprocessing rasterizes vector data to common grid; Feature engineering computes spatial statistics (mean, variance, autocorrelation) over 10 km neighborhoods; Model training uses RF or GNN on spatial features with spatial cross-validation to avoid overfitting to spatial autocorrelation; Prediction generates raster outputs showing predicted yield per grid cell.

**Results and Performance Metrics:** On 2,500 km² agricultural region (50×50 grid cells, 10 years), GIS-integrated ML achieved R² = 0.86 for pixel-level yield prediction versus R² = 0.71 for non-spatial ML models. The spatial CV methodology proved critical: standard CV inflated R² to 0.93, while spatial CV revealed true generalization (R² = 0.86).

---

#### Paper 14: Towards Precision Agriculture: Metaheuristic Model Compression

**Introduction and Background:** Precision agriculture increasingly deploys deep learning models on resource-constrained devices (edge computing, mobile phones). Model compression via pruning or quantization reduces computational requirements, but naive compression degrades accuracy. Metaheuristic optimization (genetic algorithms, differential evolution) automatically finds optimal compression configurations.

**Problem Statement:** Deep neural networks for pest/disease recognition require millions of parameters, infeasible for edge deployment. Direct pruning degrades accuracy unpredictably; manually selecting pruning ratios requires extensive experimentation. Automated metaheuristic optimization can efficiently search the vast space of possible network configurations.

**Methodology Adopted:** The approach employs Differential Evolution (DE) to optimize which neurons to prune, which kernels to reduce, and which layers to quantize. For each candidate configuration, the algorithm trains a reduced network for 5 epochs and evaluates accuracy on validation set. DE iteratively refines pruning patterns across 30 generations with population size 20.

**Results and Performance Metrics:** DE-optimized InceptionV3 with channel attention achieved 88.50% accuracy and 7.9M parameters (vs. 32MB original model), representing 99.6% parameter reduction with only 11% accuracy loss. Inference speed on mobile GPU: 45 FPS (vs. 2 FPS for full model).

---

#### Paper 15: RNN and GNN Based Prediction of Agricultural Prices with Weather Integration

**Introduction and Background:** Agricultural commodity prices reflect both weather conditions and spatial spillovers between regions. Graph neural networks model price correlations between markets, while RNNs capture temporal patterns. The combination enables forecasting that accounts for supply shocks propagating through supply chains and climate impacts on global production.

**Problem Statement:** Commodity prices exhibit regional correlations: drought in Iowa corn-belt impacts prices in Argentina and China through global supply shifts. Standard time-series models (ARIMA, LSTM) ignore these spatial linkages. The challenge is modeling how weather shocks propagate spatially through interconnected agricultural markets while maintaining temporal dependencies.

**Methodology Adopted:** The framework combines Stacked LSTM for univariate price forecasting, Spectral Temporal Graph Neural Networks (StemGNN) modeling inter-market dependencies, and Temporal Graph Convolutional Networks (T-GCN) for spatio-temporal learning. The graph connects agricultural markets (nodes) if they compete or depend on the same supply regions (edges). Weather variables are incorporated as node features.

**Results and Performance Metrics:** On commodity price forecasting (monthly corn, wheat, rice prices; 180 markets; 10 years), T-GCN achieved RMSE = 8.2% of price mean (MAE = 11.5%) versus LSTM-only RMSE = 13.4%, and StemGNN RMSE = 10.7%. Incorporating weather variables via node features reduced RMSE by additional 15%.

---

#### Paper 16: An LSTM Neural Network for Improving Wheat Yield Prediction

**Introduction and Background:** Wheat yield depends on complex temporal interactions: spring drought followed by summer flooding has different impacts than isolated drought. LSTM networks with their explicit memory units (cell state, hidden state) can learn these multi-year dependencies better than standard RNNs vulnerable to vanishing gradients.

**Problem Statement:** Wheat yield prediction requires understanding cumulative weather effects: a single-year drought is less damaging if preceded by wet years replenishing soil moisture, but catastrophic if years of drought pre-exist. Standard statistical methods cannot capture these non-linear temporal thresholds.

**Methodology Adopted:** LSTM architecture comprises input gate (controls what new information enters cell state), forget gate (controls which past information to retain), and output gate (controls what cell state information to output). The model processes 5-year weather sequences with 64 LSTM cells. Dropout (0.2) prevents overfitting. Training uses Adam optimizer (learning rate 0.001) and early stopping.

**Results and Performance Metrics:** On 29-year wheat yield data (2,500 plots across 20 provinces), LSTM achieved MAE = 178 kg/ha (test set) versus MLP MAE = 267 kg/ha, RF MAE = 234 kg/ha, and SVM MAE = 289 kg/ha. Correlation between predicted and actual yields: r = 0.94 (LSTM) vs. r = 0.81 (RF).

---

#### Paper 17: Multi-Task Transfer Learning Deep Convolutional Neural Networks

**Introduction and Background:** Multi-task learning trains a single model on multiple related tasks simultaneously, learning shared representations that improve generalization. In crop yield prediction, auxiliary tasks could include weed pressure prediction, pest risk assessment, and nutrient stress classification. These correlated tasks force the model to learn general agricultural patterns.

**Problem Statement:** Single-task learning on limited crop yield data can overfit. Multi-task learning leverages additional labeled data from related tasks to constrain the feature learning process. The shared representation must extract features useful across all tasks, improving robustness to task-specific outliers and noise.

**Methodology Adopted:** Multi-task transfer learning uses shared feature extractor (CNN layers processing all inputs, learning shared spatial patterns), task-specific heads (separate output layers for each task), and combined loss: L_total = w_yield × L_yield + w_pest × L_pest + w_disease × L_disease, with task weights balanced to prevent task imbalance issues.

**Results and Performance Metrics:** Multi-task CNN achieved better generalization than single-task CNN: on held-out data, multi-task R² = 0.89 vs. single-task R² = 0.83 (8% improvement). When yield training data was reduced by 50%, the gap widened to 15% (multi-task R² = 0.85 vs. single-task R² = 0.74).

---

#### Paper 18: Neural Regression for Scale-Varying Targets

**Introduction and Background:** Regression targets with heterogeneous scales (crop yields spanning 50-500,000 hg/ha) pose challenges: standard loss functions (MSE, MAE) fail to learn targets at extreme scales simultaneously. Autoregressive regression decomposes targets hierarchically, learning coarse-grained predictions before fine-grained adjustments.

**Problem Statement:** Global yield normalization (z-score) sometimes loses important magnitude information; task-specific normalization requires knowing target ranges a priori. Autoregressive regression targets at multiple resolutions: first predict whether yield is "very low," "low," or "high," then predict within that category.

**Methodology Adopted:** Autoregressive regression uses histogram loss: target space is discretized into buckets, and the model predicts a categorical distribution over buckets (coarse prediction) before predicting continuous values within the chosen bucket (fine prediction). For crop yields: initial buckets are {[0,100), [100,500), [500,5000), [5000,50000), [50000,∞)}.

**Results and Performance Metrics:** On multi-scale regression tasks (toy 1D with sin and log functions scaled 6 orders of magnitude apart), autoregressive regression achieved 100% accuracy on both scales versus MSE (85% accuracy on large-scale, 12% on small-scale) and MAE (78% large-scale, 18% small-scale).

---

#### Paper 19: Graph Convolution Networks Based on Adaptive Spatiotemporal Attention (GAA-GCN)

**Introduction and Background:** Spatiotemporal prediction requires learning dynamic correlations: correlations between regions vary over time (drought breaks correlations by affecting some regions differently). Adaptive spatiotemporal attention learns time-varying edge weights in graphs, enabling the network to up-weight important correlations during specific periods.

**Problem Statement:** Static graph construction assumes fixed spatial relationships, but agricultural correlations are dynamic: during drought years, moisture availability strongly correlates neighboring regions, while during wet years, temporal patterns dominate. Adaptive mechanisms allow the model to learn when to emphasize spatial vs. temporal information.

**Methodology Adopted:** GAA-GCN comprises adaptive attention mechanism (learns time-varying weights for graph edges using attention scores computed from node features), graph convolution (aggregates node features weighted by adaptive attention), LSTM layers (capture temporal dependencies in attention patterns), and fusion module (combines spatiotemporal features for prediction).

**Results and Performance Metrics:** On traffic flow prediction (urban road network, 300 sensors, 3 months), GAA-GCN achieved RMSE = 14.3% of mean traffic flow (MAE = 12.1%) versus standard GCN RMSE = 18.7%, GCN+LSTM RMSE = 16.2%, and ST-ResNet RMSE = 17.5%. The adaptive attention mechanism identified critical edge subsets.

---

#### Paper 20: A Unified Approach to Interpreting Model Predictions (SHAP)

**Introduction and Background:** Complex models like neural networks and ensemble methods lack transparency: stakeholders cannot understand why a model predicts low yield for a specific farm. SHAP (SHapley Additive exPlanations) provides principled feature importance based on game theory, explaining each prediction by quantifying each feature's marginal contribution.

**Problem Statement:** Agricultural stakeholders (farmers, policymakers) need to trust yield predictions and understand recommendations. A black-box neural network predicting "very low yield" without explanation cannot guide adaptation decisions. Explainability methods must efficiently attribute predictions to features while providing locally accurate explanations.

**Methodology Adopted:** SHAP uses Shapley values from cooperative game theory: each feature's contribution is its marginal contribution averaged over all possible feature subsets. For computational efficiency, TreeSHAP is used for tree-based models, and DeepSHAP (deep LIFT) for neural networks. SHAP provides force plots, summary plots, and dependence plots.

**Results and Performance Metrics:** On crop yield predictions (GNN model, 28,000 samples), SHAP identified top features using computationally tractable approximations (100× speedup over exact Shapley values). Feature importance rankings: rainfall (32%), temperature (25%), soil NPK (18%), neighboring-country yield (15%), pesticide use (10%) aligned with agricultural domain knowledge.

---
