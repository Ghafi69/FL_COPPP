%% CHECK_REVIEWER_CONFIG.m -- data-independent checks for revised paper config
clear; clc;
c=flcoppp_config("paper");
assert(c.fl.numRounds==60 && c.fl.numClients==50 && c.fl.clientsPerRound==10);
assert(isequal(c.statistics.seeds,[17 43 91]));
assert(c.integrity.challengeCount==8 && ~c.integrity.adaptiveProbeAllocation);
assert(numel(c.suite.attacks)==5 && numel(c.suite.methods)==7);
f=flcoppp_apply_femnist_config(c);
assert(f.data.numClasses==62 && isequal(f.data.imageSize,[28 28 1]));
assert(f.fl.numClients==50 && f.integrity.challengeCount==8);
disp('Revised reviewer configuration checks: PASS (no training performed).');
