%% RUN_SMOKE.m
% Two-round end-to-end validation. Results are not manuscript evidence.
clear; clc; close all;
cfg = flcoppp_config("smoke");
results = flcoppp_cifar10(cfg);
fprintf('\nSmoke output:\n%s\n',results.outputDir);
