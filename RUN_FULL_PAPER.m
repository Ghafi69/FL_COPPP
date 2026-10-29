%% RUN_FULL_PAPER.m -- Revised reviewer-matched 60-round CIFAR-10 experiment
clear; clc; close all;
cfg = flcoppp_config("paper");
assert(cfg.fl.numRounds==60 && ~cfg.integrity.adaptiveProbeAllocation && ...
    cfg.integrity.challengeCount==8 && numel(cfg.statistics.seeds)==3, ...
    'Manuscript profile is inconsistent with fixed-q, three-seed protocol.');
results = flcoppp_cifar10(cfg);
fprintf('\nRevised paper output folder:\n%s\n',results.outputDir);
