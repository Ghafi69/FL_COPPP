%% RUN_PAPER_WITH_ABLATIONS.m
% Runs the tractable paper_core matrix plus reviewer-facing ablations.
clear; clc; close all;

cfg = flcoppp_config("paper_core");
cfg.profile = "paper_core_with_ablations";
cfg.ablation.enabled = true;
results = flcoppp_cifar10(cfg);

fprintf('\nPaper and ablation output folder:\n%s\n',results.outputDir);
