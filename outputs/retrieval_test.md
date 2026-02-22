# retrieval_test

This file documents 3 retrieval queries against the local ChromaDB index.

At least one test demonstrates metadata filtering as required.


---

## Test 1 — EDA workflow steps

**Query:** Give me the step-by-step EDA workflow in the correct order.

**Filter (where):** `None`


### Top Results

**Hit 1** — `doc_type=eda_guideline`, `topic=eda_general`, `source=eda_handbook`

> Exploratory Data Analysis (EDA) is a critical process in the data analysis pipeline used to understand the structure, quality, and characteristics of a dataset before applying advanced modeling or drawing conclusions. The purpose of EDA is to summarize main characteristics of the data, detect patterns, identify anomalies, test assumptions, and check relationships between variables.  EDA allows analysts to gain insights into distributions, central...


**Hit 2** — `doc_type=eda_guideline`, `topic=workflow`, `source=eda_handbook`

> A structured Exploratory Data Analysis workflow ensures consistency and reliability in analytical tasks. The recommended step-by-step workflow is as follows:  Step 1: Dataset Overview Examine dataset dimensions including number of rows and columns. Identify column names and data types. Understand whether variables are numerical, categorical, or datetime.  Step 2: Data Quality Assessment Check for missing values across all columns. Identify duplic...


**Hit 3** — `doc_type=eda_guideline`, `topic=workflow`, `source=eda_handbook`

> epending on data type and distribution.  Step 5: Distribution Analysis Analyze distribution of numerical variables using histograms or density plots. Detect skewness, multimodality, or unusual spread.  Step 6: Outlier Detection Identify extreme values using statistical techniques such as Interquartile Range (IQR) or Z-score method. Evaluate whether outliers represent genuine variation or data errors.  Step 7: Correlation Analysis Compute correlat...



---

## Test 2 — Handling missing values

**Query:** How should I handle missing values during EDA? Give best practices.

**Filter (where):** `None`


### Top Results

**Hit 1** — `doc_type=eda_guideline`, `topic=missing_values`, `source=eda_handbook`

> Handling missing values is a fundamental step in exploratory data analysis. Missing data can occur due to data entry errors, system failures, or incomplete information collection.  Types of missing data include:  Missing Completely at Random (MCAR)  Missing at Random (MAR)  Missing Not at Random (MNAR)  Common strategies for handling missing values include:  Deletion Methods: Removing rows or columns with missing values when the proportion is sma...


**Hit 2** — `doc_type=eda_guideline`, `topic=eda_general`, `source=eda_handbook`

> Exploratory Data Analysis (EDA) is a critical process in the data analysis pipeline used to understand the structure, quality, and characteristics of a dataset before applying advanced modeling or drawing conclusions. The purpose of EDA is to summarize main characteristics of the data, detect patterns, identify anomalies, test assumptions, and check relationships between variables.  EDA allows analysts to gain insights into distributions, central...


**Hit 3** — `doc_type=eda_guideline`, `topic=workflow`, `source=eda_handbook`

> A structured Exploratory Data Analysis workflow ensures consistency and reliability in analytical tasks. The recommended step-by-step workflow is as follows:  Step 1: Dataset Overview Examine dataset dimensions including number of rows and columns. Identify column names and data types. Understand whether variables are numerical, categorical, or datetime.  Step 2: Data Quality Assessment Check for missing values across all columns. Identify duplic...



---

## Test 3 — Metadata filtering (ONLY correlation topic)

**Query:** Explain correlation analysis and how to interpret correlation strength.

**Filter (where):** `{'topic': 'correlation'}`


### Top Results

**Hit 1** — `doc_type=eda_guideline`, `topic=correlation`, `source=eda_handbook`

> Correlation analysis measures the strength and direction of relationships between numerical variables. It helps identify patterns and dependencies that may inform predictive modeling or decision-making.  Pearson Correlation: Measures linear relationship between two continuous variables. Values range from −1 to +1.  Interpretation: +1 indicates perfect positive relationship 0 indicates no linear relationship −1 indicates perfect negative relations...

