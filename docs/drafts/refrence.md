# Enhanced Crop Yield Prediction Using a Hybrid ANN–COA Model

## Abstract

Climate change and global population growth intensify food security challenges, necessitating accurate and reliable crop yield forecasting models. This work presents a hybrid prediction framework that integrates an Artificial Neural Network (ANN) with a Levy flight–enhanced Coati Optimization Algorithm (COA). The proposed model incorporates climatic variables, pesticide usage, and historical crop yield data across ten major crops and 102 countries. Experimental evaluation demonstrates that the ANN–COA model significantly outperforms baseline ANN and other hybrid metaheuristic-based models, achieving high prediction accuracy with strong generalization capability. The results confirm the effectiveness of swarm-based optimization and Levy flight mechanisms in addressing nonlinearity and high-dimensionality in agricultural datasets.

---

## Keywords

Crop Yield Prediction · Artificial Neural Network · Coati Optimization Algorithm · Levy Flight · Metaheuristic Optimization · Precision Agriculture

---

## 1. Introduction

Accurate crop yield forecasting is essential for agricultural planning, food security strategies, and policy-making. Traditional yield estimation approaches, often based on empirical methods or farmer experience, lack scalability and adaptability to dynamic climatic conditions. Crop yield prediction is inherently complex due to nonlinear interactions among climatic, biological, and management-related variables.

Recent advances in Artificial Intelligence (AI) and Machine Learning (ML), particularly Artificial Neural Networks (ANNs), have enabled improved modeling of nonlinear agricultural processes. However, conventional ANN training methods frequently suffer from slow convergence, sensitivity to initialization, and entrapment in local optima. To mitigate these limitations, swarm intelligence–based optimization algorithms have been introduced to improve ANN parameter tuning.

This study proposes a novel hybrid framework combining ANN with a Levy flight–enhanced Coati Optimization Algorithm (COA) to enhance prediction accuracy and robustness in large-scale crop yield forecasting.

---

## 2. Related Work

Previous research in crop yield prediction has explored regression models, ensemble learning, decision trees, support vector machines, and deep learning architectures such as CNNs and LSTMs. Hybrid approaches that combine neural networks with metaheuristic algorithms (e.g., PSO, GWO, WOA) have demonstrated improved convergence behavior and predictive performance.

Despite these advancements, challenges persist in balancing exploration and exploitation, avoiding premature convergence, and handling highly nonlinear, multi-variate agricultural datasets. This work addresses these gaps through a novel ANN–COA hybrid model enhanced with Levy flight dynamics.

---

## 3. Materials and Methods

### 3.1 Data Collection

The dataset was obtained from FAOSTAT and the World Data Bank, covering annual records from 2000 to 2013 across 102 countries. Ten major crops were considered:

- Cassava  
- Maize  
- Plantains  
- Potatoes  
- Rice (Paddy)  
- Sorghum  
- Soybeans  
- Sweet Potatoes  
- Yam  
- Wheat  

The dataset includes crop yield values along with climatic and agronomic variables.

---

### 3.2 Data Preprocessing

Categorical variables such as Nation and Crop Type were converted into numerical representations using label encoding. The final feature set includes:

- Nation  
- Crop_Type  
- Production_Year  
- Yield (hg/ha)  
- Mean_Annual_Rainfall (mm)  
- Pesticide_Application (tonnes)  
- Average_Temperature (°C)  

All numerical variables were normalized using min–max normalization to ensure uniform scaling. The dataset was split into training (70%) and testing (30%) subsets.

---

## 4. Proposed Methodology

### 4.1 Artificial Neural Network (ANN)

The ANN serves as a nonlinear regression model that estimates crop yield from the input feature set. The network consists of an input layer, a hidden layer with sigmoid activation, and an output layer representing yield. Mean Square Error (MSE) is used as the loss function.

---

### 4.2 Coati Optimization Algorithm (COA)

COA is a population-based metaheuristic inspired by the cooperative hunting and predator evasion behaviors of coatis. It operates in two main phases:

- **Exploration phase**: Models cooperative attack strategies to explore the global search space.  
- **Exploitation phase**: Simulates predator evasion to refine candidate solutions locally.

---

### 4.3 Levy Flight Enhancement

Levy flight dynamics are integrated into COA to introduce probabilistic long-distance jumps during optimization. This mechanism improves global search capability and prevents premature convergence by enabling escape from local optima.

---

### 4.4 Hybrid ANN–COA Framework

In the hybrid framework, COA optimizes ANN weights and biases by minimizing the Mean Square Error fitness function. The optimization process iteratively updates ANN parameters until convergence or a predefined iteration limit is reached. This integration significantly enhances training efficiency and prediction accuracy.

---

## 5. Performance Evaluation Metrics

Model performance was evaluated using the following statistical metrics:

- Root Mean Square Error (RMSE)  
- Mean Absolute Error (MAE)  
- Mean Absolute Percentage Error (MAPE)  
- Coefficient of Determination (R²)  

These metrics quantify both absolute and relative prediction accuracy and assess generalization capability.

---

## 6. Results and Discussion

### 6.1 Comparative Model Performance

The proposed ANN–COA model was evaluated against baseline ANN and hybrid ANN models optimized using PSO, GWO, and WOA. The ANN–COA model consistently achieved the best performance across all metrics.

**Key Results (Testing Phase):**
- RMSE: 15,534.5  
- MAE: 10,425.7  
- MAPE: 0.07794  
- R²: 0.96845  

These results demonstrate that the ANN–COA framework provides superior predictive accuracy and robustness compared to other methods.

---

### 6.2 Ablation and Feature Importance Analysis

A systematic feature exclusion study was conducted to assess variable importance. Results indicate:

- Climatic variables (Mean Annual Rainfall and Average Temperature) are the most critical contributors to yield prediction accuracy.
- Crop_Type is the most influential biological factor.
- Nation and Production_Year exhibit moderate influence.
- Pesticide_Application has the least impact on prediction accuracy.

This analysis improves model interpretability and informs future feature prioritization.

---

### 6.3 Comparison with Deep Learning and Time-Series Models

The ANN–COA model was further compared with LSTM, CNN–RNN, and ARIMA models across multiple crop types. The proposed model consistently achieved higher R² values and lower error metrics, particularly for staple crops such as rice and wheat, demonstrating its suitability for real-world agricultural forecasting.

---

## 7. Conclusion

This study introduces a hybrid ANN–COA framework for crop yield prediction that effectively addresses the challenges of nonlinear, high-dimensional agricultural data. The integration of swarm intelligence and Levy flight mechanisms enables robust global optimization of ANN parameters, resulting in superior prediction accuracy and convergence stability. The proposed model offers strong potential for deployment in agricultural decision-support systems and food security planning.

Future work will focus on incorporating remote sensing data, advanced deep learning architectures, and real-time forecasting capabilities.

---

## Funding

No funding was received for this work.

---

## Data Availability

The dataset used in this study is publicly available from FAOSTAT and Kaggle.

---