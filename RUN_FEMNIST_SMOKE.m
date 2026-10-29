%% RUN_FEMNIST_SMOKE.m -- Two-round integration test, NOT paper evidence
clear; clc; close all;
cfg=flcoppp_config("femnist_smoke");
results=flcoppp_cifar10(cfg);
fprintf('\nFEMNIST smoke outputs: %s\n',results.outputDir);
