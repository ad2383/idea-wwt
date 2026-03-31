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