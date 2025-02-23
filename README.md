# solid-octo-fiesta
Comparing Classifiers
# Comparing Classifiers

## Overview
This project focuses on predicting whether a client will subscribe to a term deposit (bank marketing dataset) by comparing multiple classification models. The notebook explores data preprocessing, exploratory data analysis (EDA), model training, hyperparameter tuning, and final recommendations based on various performance metrics (accuracy, precision, recall).

**Link to the Notebook:** [https://github.com/nc007-cloud/solid-octo-fiesta/blob/main/comparing_classifiers.ipynb]  
**

---

## Data and Preprocessing
- **Datasets:**  Dataset for this analysis were downloaded from the UC Irvine Machine Learning repositor. The data can be found here: https://archive.ics.uci.edu/dataset/222/bank+marketing
  - `bank-additional-full.csv`, `bank-additional.csv`  
  - `bank-full.csv`, `bank.csv`
- **Shape:**  
  - Primary dataset (`bank-full.csv`) has **41,188** rows.
  - Secondary dataset (`bank-additional.csv`) is smaller and used for computationally expensive tasks (e.g., hyperparameter tuning).
- **Missing Values:**  
  - There are **0** missing values in each dataset.
- **Dropped Columns:**  
  - `pdays`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed` were removed due to limited predictive value.

---

## Exploratory Data Analysis (EDA)
1. **Numeric Variables**  
   - **Age**: Average is ~40, range 17–98 (diverse client base).  
   - **Duration**: Average call ~4 minutes; longer calls often correlate with higher subscription likelihood.  
   - **Campaign**: Median ~2 contacts per client, max can go up to 50.  
2. **Categorical Variables**  
   - **Month**: May is most common, indicating possible seasonality.  
   - **Day of Week**: Fairly even distribution.  
   - **Contact Type**: ~63% cellular vs. landline.  
3. **Key Observations**  
   - Success rate from previous campaigns ~3%, showing a class imbalance (many more “no” than “yes”).  
   - Age is heavily centered between 30 and 50.  
   - Most calls are relatively short (<6 minutes).

---

## Modeling Approach
1. **Train/Test Split**  
   - Data partitioned into training and test sets for robust evaluation.
2. **Models Explored**  
   - **Decision Tree** (Default & Tuned)  
   - **Random Forest** (Default & Tuned)  
   - **Logistic Regression**  
   - **K-Nearest Neighbors**  
   - **Support Vector Machine (SVM)**
3. **Hyperparameter Tuning**  
   - **GridSearchCV** used for Decision Tree & Random Forest on the smaller `bank-additional` dataset to manage computational cost.  
   - Tuned parameters include tree depth, class weights, and number of estimators.

---

## Model Performance and Key Findings

| Model                  | Accuracy   | Precision (Class=1) | Recall (Class=1)  | Notes                                                                  |
|------------------------|-----------:|---------------------:|-------------------:|-------------------------------------------------------------------------|
| **Decision Tree**      | ~89%       | ~0.69               | ~0.71             | Overfits on training data (100% recall), less generalizable            |
| **Decision Tree Tuned**| ~88%       | **0.82**            | ~0.65             | Highest **precision**, fewer false positives, but misses some positives|
| **Random Forest**      | **90.87%** | 0.79                | 0.69              | Best **overall balance** of precision & recall                         |
| **Random Forest Tuned**| ~90%       | ~0.83               | **0.70**          | Highest **recall**, captures more positives but more false positives   |
| **Logistic Regression**| ~88.8%     | 0.68                | 0.62              | Performs reasonably well, simpler model                                |
| **SVM**                | ~86–87%    | Lower               | Lower             | Struggles with this dataset                                            |

### Business Implications
- **High Precision (Decision Tree Tuned)**  
  - Minimizes false positives; useful if marketing costs per contact are high.  
- **High Recall (Random Forest Tuned)**  
  - Minimizes false negatives; useful if missing potential subscribers is very costly.  
- **Balanced Approach (Random Forest Default)**  
  - Offers a good trade-off between precision and recall, making it an excellent all-around choice.

---

## Recommendations
1. **Select Model Based on Business Priorities**  
   - If you need to minimize wasted outreach, choose **Decision Tree Tuned** (high precision).  
   - If you need to capture as many subscribers as possible, choose **Random Forest Tuned** (high recall).  
   - For a balanced strategy, **Random Forest (Default)** is ideal.
2. **Next Steps**  
   - **Feature Engineering**: Explore additional features (e.g., digital interactions) to boost predictive power.  
   - **Threshold Adjustment**: Fine-tune classification thresholds to optimize precision vs. recall.  
   - **Further Validation**: Test on new/external data to confirm performance.  
   - **Deployment**: Implement the chosen model and monitor metrics in a live environment.

---

## Project Organization
- **Notebook**  
  - Clearly structured with sections: Data Loading, EDA, Modeling, Hyperparameter Tuning, and Conclusions.  
  - Code cells are annotated with explanatory comments.
- **Data**  
  - CSV files are stored in a dedicated folder (e.g., `data/` or `bank+marketing/`).  
  - No unnecessary files are included in the repository.
- **Dependencies**  
  - Python 3.x, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, and Jupyter Notebook.

---

## How to Reproduce
1. **Clone or Download** the repository containing the data and notebook.  
2. **Install Dependencies** (e.g., `pip install -r requirements.txt`) or manually install the libraries.  
3. **Open** the `Comparing_Classifiers.ipynb` in Jupyter or a compatible environment.  
4. **Run** the cells in order to replicate the analysis and results.

---

**Thank you for reviewing this project!** If you have any questions or suggestions, please feel free to reach out.
