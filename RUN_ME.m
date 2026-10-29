%% RUN_ME.m
% Recommended first reviewer-facing run.
% 1) Edit cfg.paths.cifarRoot in flcoppp_config.m.
% 2) Run RUN_PROTOCOL_TESTS.m.
% 3) Run RUN_SMOKE.m.
% 4) Run this script.
clear; clc; close all;

profile = "paper_core"; % change to "paper" only after paper_core is validated
cfg = flcoppp_config(profile);
results = flcoppp_cifar10(cfg);

fprintf('\nReviewer-facing output folder:\n%s\n',results.outputDir);
