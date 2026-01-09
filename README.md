# MACHINE LEARNING PROJECTS - FINAL SUBMISSION

**Authors:** Rajat Dogra & Umair Aziz  
**Course:** ML2 - Machine Learning 2  
**Date:** January 10, 2026  
**University:** University of Warsaw, Faculty of Economic Sciences

## PROJECT OVERVIEW

This submission contains two comprehensive machine learning projects demonstrating advanced techniques in both regression and classification problems.

## PROJECT 1: REGRESSION - Cricket Score Prediction

- **Problem:** Predict IPL cricket match scores
- **Dataset:** IPL Cricket Data from Kaggle (76,014 records)
- **Target:** Continuous score values
- **Algorithms:** 8+ regression models (Linear, Ridge, Lasso, Random Forest, XGBoost, etc.)
- **Best Result:** Random Forest with R² = 0.89, RMSE = 12.4 runs
- **Key Features:** Ball-by-ball analysis, team performance, venue effects

## PROJECT 2: CLASSIFICATION - Football Match Outcomes

- **Problem:** Predict football match outcomes (Home Win/Draw/Away Win)
- **Dataset:** ESPN Soccer Data from Kaggle (67,353 matches)
- **Target:** 3-class categorical outcome
- **Algorithms:** 5+ classification models (Random Forest, XGBoost, Neural Networks, Ensembles)
- **Best Result:** Stacking Ensemble with 65.36% accuracy, F1 = 0.6533
- **Key Features:** Team historical performance, league effects, temporal patterns

## SUBMISSION STRUCTURE

```
final_submission/
├── classification/                    # Football Match Outcome Classification
│   ├── football_classification_complete.py    # Main Python script
│   ├── results/                              # All results and analysis
│   │   ├── comprehensive_results.txt         # Main results output
│   │   ├── error_analysis.txt               # Error analysis report
│   │   └── ethical_considerations.txt       # Ethical considerations
│   ├── plots/                               # Professional visualizations
│   │   ├── model_performance_comparison.png  # Model comparison charts
│   │   ├── per_class_performance.png        # Per-class analysis
│   │   └── feature_importance.png           # Feature importance plot
│   └── data/                                # Dataset
│       └── fixtures.csv                     # ESPN Soccer Data
├── regression/                        # Cricket Score Prediction
│   ├── Cricket_Score_Prediction_Final_Submission.ipynb  # Jupyter notebook
│   ├── Cricket_Score_Prediction_Final_Submission.pdf   # PDF output
│   └── regression_dataset.csv                          # IPL Cricket Data
├── presentation/                      # Combined presentation
│   ├── ml_projects_presentation.tex   # LaTeX source
│   ├── wne-logo-new-en.png          # University logo
│   └── fig_wfo.png                   # Supporting figures
```

## DATASETS USED

### 1. Cricket Dataset
- **Source:** [Kaggle - IPL Dataset 2008-2025](https://www.kaggle.com/datasets/chaitu20/ipl-dataset2008-2025)
- **Description:** IPL matches from 2008-2025
- **Details:** Ball-by-ball granular data
- **Size:** 76,014+ records with 60+ features

### 2. Football Dataset
- **Source:** [Kaggle - ESPN Soccer Data](https://www.kaggle.com/datasets/excel4soccer/espn-soccer-data)
- **Description:** ESPN Soccer matches from 2024-2026
- **Details:** Match-level data across multiple leagues
- **Size:** 67,353 matches with comprehensive metadata

## RESULTS SUMMARY

### Regression (Cricket)
- **Best Model:** Random Forest
- **R² Score:** 0.89
- **RMSE:** 12.4 runs
- **Key Insight:** Current run rate and wickets most predictive

### Classification (Football)
- **Best Model:** Stacking Ensemble
- **Accuracy:** 65.36%
- **F1-Score:** 0.6533
- **Key Insight:** Team strength difference most important feature

## ACKNOWLEDGMENTS

- Kaggle community for high-quality datasets
- scikit-learn, pandas, matplotlib development teams
- University of Warsaw ML2 course materials and guidance
