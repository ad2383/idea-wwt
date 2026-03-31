%% WWTP Modeling
% Runs all 4 phases end-to-end.
% Each phase is also independently runnable.

% Phase 1: Data Preparation
disp('=== Phase 1: Data Preparation ===');
src/loadAndPrepareData

% Phase 2: Training & Timing
disp('=== Phase 2: Training & Timing ===');
src/trainAllModels

% Phase 3: Evaluation
disp('=== Phase 3: Evaluation ===');
src/evaluateModels

% Phase 4: Visualization & Normalization
disp('=== Phase 4: Visualization & Normalization ===');
src/visualizeResults
src/normalizeAndRetrain

disp('=== All phases complete ===');
