# EEG Artifact Rejection

The Jupyter notebook(s) and script(s) in the *EEG-artifact-rejection* repository are prototypes for a more automated process in detecting and rejecting EEG artifacts.

## Usage

This ongoing process explores the efficacy of using SVM and Isolation Forest to detect artifacts in EEG data and possibly other time series.

## Background Information
### Datasets
  - Unfortunately, the datasets used for training and testing cannot be publicly released due to privacy reasons.
  - The prototype is currently in development for [The Walker Sleep and Neuroimaging Lab (Center for Human Sleep Science)](https://vcresearch.berkeley.edu/research-unit/center-human-sleep-science), led by [Professor Matthew Walker](https://vcresearch.berkeley.edu/faculty/matthew-walker), at UC Berkeley. Please reach out to Matthew Walker for any questions, comments, and/or concerns.

### Possible Approaches
  - Isolation Forest
    - [Isolation Forest](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
  - SVM Classifier or SVC
  - Other comprehensive approaches:
    - [A Review on Machine Learning Algorithms in Handling EEG Artifacts](http://www.es.mdh.se/pdf_publications/3562.pdf)
    - [Automated EEG artifact elimination by applying machine learning algorithms to ICA-based features](http://iopscience.iop.org/article/10.1088/1741-2552/aa69d1/meta)

### Key Challenges
  - The labels for artifacts are only mapped per epoch, but the sampling rate produces up to hundreds of data points per epoch.
  - Depending on the chosen algorithm, feature selection process, etc., the machine learning model might take more than a *reasonable* amount of RAM and time just for the training dataset.
  - The target value for the [recall score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html) >= 0.85 while maintaining a [precision score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score) > 0.60.
    - Even though detecting and removing artifacts are extremely important, the rejection process should not jeopardize signals of interest.
