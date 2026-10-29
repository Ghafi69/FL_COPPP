%% RUN_REVIEWER_COMPRESSION.m
% Clean-trace matched ablation, three seeds and 60 rounds. Results are new
% experimental evidence after execution, not a reconstruction of R1 data.
clear;clc;close all;
cfg=flcoppp_config("paper");
cfg.profile="reviewer_compression_60round";
cfg.suite.attacks="clean";
cfg.suite.methods=["FedAvg";"FLCOPPP"; ...
    "FLCOPPP_NoCompression";"FLCOPPP_NoPreGate"];
results=flcoppp_cifar10(cfg);
fprintf('Compression attribution results: %s\n',results.outputDir);
