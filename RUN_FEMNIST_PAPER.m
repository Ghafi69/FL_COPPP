%% RUN_FEMNIST_PAPER.m -- LEAF writer-based FEMNIST, 60 rounds x 3 seeds
clear; clc; close all;
cfg = flcoppp_config("paper");
cfg = flcoppp_apply_femnist_config(cfg);
assert(cfg.fl.numRounds==60 && cfg.integrity.challengeCount==8 && ...
    ~cfg.integrity.adaptiveProbeAllocation, 'FEMNIST settings mismatch.');
results = flcoppp_cifar10(cfg);
fprintf('\nFEMNIST outputs saved to:\n%s\n',results.outputDir);
