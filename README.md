# Data-driven Models for Wastewater Treatment Plant

A MATLAB project that uses supervised learning to model and predict wastewater treatment plant outputs from sensor data.

## Overview

We train and compare 7 supervised learning algorithms on real WWTP sensor data to predict water quality metrics like dissolved oxygen, nitrate, and ammonium concentrations.


```mermaid
flowchart TD
    A([Start]) --> B

    subgraph P1["Phase 1 — Learn the framework"]
        B["Run example scripts\nwith energyDataset.mat"]
        B --> C["Understand fitlm, predict,\nand MSE calculation"]
    end

    C --> D

    subgraph P2["Phase 2 — Prepare data"]
        D["Load waterTreatmentData.mat"]
        D --> E["Build 7-column input matrix"]
        E --> F["80/20 train/test split\nusing randperm"]
    end

    F --> G

    subgraph P3["Phase 3 — Train and evaluate"]
        G["Train 7 algorithms\non each output variable"]
        G --> H["Calculate RMSE\non test data"]
        H --> I{All 5 outputs done?}
        I -->|No| G
        I -->|Yes| J["7×5 RMSE comparison table"]
    end

    J --> K

    subgraph P4["Phase 4 — Analyze"]
        K["Compare models\nacross all outputs"]
        K --> L["Visualize model fits\non test data"]
        L --> M["Normalize data\nand retrain"]
    end

    M --> N([Best model recommendation])
```

## Algorithms Used

- Linear Regression (`fitlm`)
- Generalized Linear Regression (`fitglm`)
- Gaussian Process Regression (`fitrgp`)
- Support Vector Machine (`fitrsvm`)
- Decision Tree (`fitrtree`)
- Ensemble of Learners (`fitrensemble`)
- Generalized Additive Model (`fitrgam`)

## Inputs and Outputs

**Inputs:** `Q_inf`, `Q_air_1`, `Q_air_2`, `Q_air_3`, `Q_air_4`, `Q_air_5`, `Temp`

**Outputs:** `DO_1`, `DO_2`, `DO_3`, `NO3`, `NH4`

## Team
Soham Bhagat, Jason Ta, Alona Dhal