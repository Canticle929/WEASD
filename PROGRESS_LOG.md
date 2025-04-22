# Project Progress Log / 项目进展日志

**Last Updated / 最后更新时间:** 2024-04-20 

## Phase 1: Project Setup & Planning / 阶段一：项目设置与规划

*   **Project Initialization / 项目初始化:**
    *   Initialized Git repository.
        *   初始化了 Git 仓库。
    *   Created `.gitignore` file to exclude data and temporary files.
        *   创建了 `.gitignore` 文件以排除数据和临时文件。
    *   Created `LICENSE` (MIT) and `requirements.txt` files.
        *   创建了 `LICENSE` (MIT) 和 `requirements.txt` 文件。
    *   Created a basic `README.md` file.
        *   创建了基础的 `README.md` 文件。

*   **Project Planning / 项目规划:**
    *   Defined main phases: Data Preparation (cleaning, windowing, optional normalization), Model Development (feature engineering, model selection, training/evaluation), Code Integration, Version Control.
        *   明确了主要步骤：数据准备（清理、窗口化、归一化可选）、模型开发（特征工程、模型选择、训练评估）、代码整合、版本控制。

## Phase 2: Data Preparation / 阶段二：数据准备

*   **Data Cleaning / 数据清理:**
    *   Selected `data_cleaning.py` as the main data cleaning script.
        *   选择了 `data_cleaning.py` 作为主要的数据清理脚本。
    *   Debugged and fixed issues in `data_cleaning.py` (incorrect raw data path, `KeyError` on sampling rate retrieval).
        *   调试并修正了 `data_cleaning.py` 中的问题（原始数据路径错误、采样率获取 `KeyError`）。
    *   Successfully executed `data_cleaning.py`. Cleaned data (ECG, EDA, Resp signals, labels 1 & 2) saved in `/Volumes/xcy/TeamProject/WESAD/cleaned_data`, with subdirectories for each subject, along with statistical reports and visualizations.
        *   成功执行了 `data_cleaning.py`。清理后的数据（ECG, EDA, Resp 信号，标签 1 和 2）已保存在 `/Volumes/xcy/TeamProject/WESAD/cleaned_data` 目录下，每个受试者一个子目录，并生成了统计报告和可视化图表。
*   **Data Windowing / 数据窗口化:**
    *   Selected `simple_windowing.py` for windowing, using a 5s window size, 2.5s step (50% overlap), and mode for window labeling.
        *   选择了 `simple_windowing.py` 作为数据窗口化脚本，采用 5 秒窗口大小和 2.5 秒步长（50% 重叠），窗口标签使用众数确定。
    *   Successfully executed `simple_windowing.py`. Windowed data saved in `/Volumes/xcy/TeamProject/WESAD/windowed_data`, including individual subject windows and merged data in the `merged` subdirectory. A total of 11,012 windows were generated.
        *   成功执行了 `simple_windowing.py`。窗口化数据保存在 `/Volumes/xcy/TeamProject/WESAD/windowed_data` 目录下，包含每个受试者的窗口数据和合并后的数据 (`merged` 子目录)。共生成 11012 个窗口。

## Phase 3: Model Development / 阶段三：模型开发

*   **Feature Engineering / 特征工程:**
    *   Created `feature_extraction.py` to extract statistical features from windowed signals.
        *   创建了 `feature_extraction.py` 脚本用于从窗口化信号中提取统计特征。
    *   Implemented extraction of 17 time-domain features for each signal type (ECG, EDA, Resp), including basic statistics (mean, std, range, etc.), shape features (skewness, kurtosis), variability measures (RMS, zero-crossings), and rate-of-change metrics.
        *   为每种信号类型（ECG, EDA, Resp）实现了 17 种时域特征的提取，包括基础统计特征（均值、标准差、范围等）、形态特征（偏度、峰度）、变异性指标（均方根、过零点数）和变化率指标。
    *   Added visualization capabilities to examine feature distributions and correlations.
        *   添加了可视化功能，用于检查特征分布和相关性。

## Next Steps / 下一步

*   **Model Development - Model Training:** Create machine learning models to classify baseline vs. stress states using the extracted features. Consider a variety of classifiers (e.g., Logistic Regression, Random Forest, SVM).
    *   **模型开发 - 模型训练:** 使用提取的特征创建机器学习模型，用于分类基线与压力状态。考虑多种分类器（如逻辑回归、随机森林、SVM）。
*   **Model Development - Evaluation:** Implement cross-validation and performance metrics to assess model quality.
    *   **模型开发 - 评估:** 实现交叉验证和性能指标，用于评估模型质量。 