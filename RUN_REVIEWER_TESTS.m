%% RUN_REVIEWER_TESTS.m -- config and protocol tests; no dataset needed
clear;clc;
c=flcoppp_config("paper");
assert(c.fl.numRounds==60 && c.integrity.challengeCount==8);
assert(~c.integrity.adaptiveProbeAllocation);
assert(numel(c.statistics.seeds)==3);
assert(c.data.validationCount>=c.behavior.referenceCount);
f=flcoppp_apply_femnist_config(c);
assert(f.data.numClasses==62 && isequal(f.data.imageSize,[28 28 1]));
assert(f.fl.numClients==50 && f.fl.clientsPerRound==10);
% Exact hypergeometric detection probability for N=97, m=5, q=8.
p=flcoppp_protocol("exactdetection",97,5,8);
assert(abs(p-0.35593)<0.0001,'Unexpected exact hypergeometric result.');
% Protocol independent of datasets and GPU.
report=flcoppp_protocol("selftest");
assert(report.Passed,'Cryptographic self-test failed.');
disp('Reviewer configuration, probability and cryptographic tests PASSED.');
