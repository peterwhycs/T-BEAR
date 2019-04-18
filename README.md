# Time-Based EEG Artifact Rejection (T-BEAR)

Automated process for detecting and rejecting EEG artifacts.

## Table of Contents

## Description

This ongoing process explores supervised learning methods to detect artifacts in EEG data and possibly other time series.

### Challenges:
  - Current models are often task specific
  - Feature engineering & selection
  - High dimensionality
  - High variability between subjects
  - Low signal-to-noise ratio
  - Non-stationary signal

### Possible Prototypes:
  - **Machine Learning**
    - Supervised:
      - Random Forest Classifier/Regressor
      - Support Vector Classifier/Machine

    - Unsupervised:
      - Isolation Forest

  - **Deep Learning**
    - Supervised:
      - Convolutional Neural Network (CNN)*
      - Recurrent Neural Network (RNN)

\*CNN will be our goal for the final model.

### Performance Metrics:
  - F1 Score
  - Precision
  - Recall
  - AUC-ROC Curve

## Getting Started

### Dependencies

- **Anaconda**: within the `tbear` directory containing the file `environment.yml` perform:
  - Problems may arise with Windows users.

```
conda env create -f environment.yml
```


- **pip**

```
pip install numpy scipy matplotlib pandas scikit-learn jupyter mne
```

<!-- ### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders -->

<!-- ### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
``` -->

<!-- ## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
``` -->
<!--
## Authors

Contributors names and contact info

ex. Dominique Pizzie
ex. [@DomPizzie](https://twitter.com/dompizzie) -->

<!-- ## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release -->

## License

This project is licensed under the Apache License - see the [LICENSE](./LICENSE) file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck, L. Parkkonen, M. Hämäläinen, MNE software for processing MEG and EEG data, NeuroImage, Volume 86, 1 February 2014, Pages 446-460, ISSN 1053-811](https://martinos.org/mne/stable/index.html)
* [Roy, Yannick & Banville, Hubert & Albuquerque, Isabela & Gramfort, Alexandre & Faubert, Jocelyn. (2019). Deep learning-based electroencephalography analysis: a systematic review.](https://arxiv.org/pdf/1901.05498.pdf)
