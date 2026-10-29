%% RUN_PROTOCOL_TESTS.m
% Run this first. It does not need CIFAR-10 or the Deep Learning Toolbox.
clear; clc;

cfg = flcoppp_config("protocol_test");
results = flcoppp_cifar10(cfg);
assert(results.ProtocolSelfTest.Passed,'Protocol tests failed.');
disp(results.ProtocolSelfTest);
fprintf('\nAll separated client/server protocol tests passed.\n');
