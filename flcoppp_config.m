function cfg = flcoppp_config(profile)
%FLCOPPP_CONFIG Reviewer-facing configuration for the FL-COP++ suite.
%
% Profiles
%   protocol_test         cryptographic/protocol unit tests only
%   smoke                 two-round end-to-end pipeline validation
%   paper_core            tractable main experiment for first paper run
%   paper                 full reviewer-facing main experiment
%   paper_with_ablations  full main experiment plus component/sensitivity runs
%
% The CIFAR path may point to either the parent folder or directly to
% cifar-10-batches-mat.

if nargin < 1
    profile = "smoke";
end
profile = string(profile);

%% Data paths. Set environment variables, or edit these two paths locally.
% Outputs MUST be independent of dataset installation (data may be read-only).
cifarEnv = string(getenv('FLCOPPP_CIFAR_ROOT'));
femnistEnv = string(getenv('FLCOPPP_FEMNIST_ROOT'));
if strlength(cifarEnv)==0, cifarEnv=fullfile(pwd,'data','cifar10'); end
if strlength(femnistEnv)==0, femnistEnv=fullfile(pwd,'data','femnist'); end
cfg.paths.cifarRoot = cifarEnv;
cfg.paths.femnistRoot = femnistEnv;
cfg.paths.outputRoot = fullfile(pwd,'FLCOPPP_outputs_revised');

%% Reproducibility and execution
cfg.profile = profile;
cfg.datasetName = "CIFAR-10";
cfg.execution.useGPU = true;
cfg.execution.verbose = true;
cfg.execution.resume = true;
cfg.execution.saveCheckpoints = true;
cfg.execution.checkpointEveryRounds = 5;
cfg.execution.checkpointSchemaVersion = 6; % invalidate incompatible pre-revision checkpoints
cfg.execution.runProtocolTestsBeforeSuite = true;

%% Data and model
cfg.data.validationCount = 1000; % assumed clean trusted SERVER-HELD reference set
cfg.data.referenceSetAssumption = "trusted disjoint subset of public training split; never used in client updates";
cfg.data.femnistMinUserSamples = 40; % writers below this train count are ineligible
cfg.data.femnistMaxSamplesPerWriter = inf; % no hidden caps in data loading
cfg.data.maxTrainSamplesPerClientRound = 512;
cfg.data.numClasses = 10;
cfg.data.imageSize = [32 32 3];
cfg.data.normalize = true;
% Public CIFAR-10 normalization constants; no server access to the full
% federated training pool is required to estimate them.
cfg.data.channelMean = [0.4914 0.4822 0.4465];
cfg.data.channelStd = [0.2470 0.2435 0.2616];
cfg.data.normalizationSource = "public CIFAR-10 constants";
cfg.model.name = "CompactCIFARConvNet";
cfg.model.dropout = 0.10;

%% Federated learning
cfg.fl.numClients = 50;
cfg.fl.clientsPerRound = 10;
cfg.fl.numRounds = 60; % R1 study horizon; no silent 100-round default
cfg.fl.localEpochs = 1;
cfg.fl.miniBatchSize = 64;
cfg.fl.learnRate = 0.02;
cfg.fl.learnRateDecay = 0.995;
cfg.fl.gradientClip = 5.0;
cfg.fl.nonIIDAlpha = 0.5;
cfg.fl.minSamplesPerClient = 100;
cfg.fl.evalEvery = 5;

%% Threat model and attacks
cfg.attack.maliciousFraction = 0.20;
cfg.attack.startRound = 5;
cfg.attack.poisonFraction = 0.30;
cfg.attack.sourceLabel = 1;
cfg.attack.targetLabel = 7;
cfg.attack.backdoorPatchSize = 4;
cfg.attack.backdoorValue = uint8(255);
cfg.attack.modelReplacementScale = 0; % 0 = derive from aggregation participation
cfg.attack.signFlipScale = 5.0;
cfg.attack.targetLayerNames = ["fc"];

cfg.suite.attacks = [
    "clean"
    "labelFlipAfterCommit"
    "backdoorBeforeCommit"
    "modelReplacementBackdoor"
    "signFlip"
    "targetedLayerReplacement"
];

cfg.suite.methods = [
    "FedAvg"
    "CoordinateMedian"
    "TrimmedMean"
    "Krum"
    "MultiKrum"
    "RFA"
    "FLTrust"
    "PreGateOnly"
    "FLCOPPP"
];

%% Integrity protocol
cfg.integrity.enabled = true;
cfg.integrity.blockSize = 4096;
cfg.integrity.challengeCount = 8;
cfg.integrity.adaptiveProbeAllocation = false; % fixed primary q=8, Eq.(9) hypergeometric
cfg.integrity.probePolicy = "fixed_uniform_without_replacement";
cfg.integrity.minimumChallengesPerClient = 2;
cfg.integrity.secondaryChallengeMultiplier = 4;
cfg.integrity.nonceBytes = 16;
cfg.integrity.aeadNonceBytes = 12;
cfg.integrity.aeadTagBytes = 16;
cfg.integrity.microbenchTrials = 500;
cfg.integrity.partialKnowledgeFraction = 0.50;
cfg.integrity.baselineChallengeCounts = [2 4 8 16 32];
cfg.integrity.baselineTamperFractions = [0.005 0.01 0.05 0.10];
cfg.integrity.baselineDataFractions = [0.25 0.50 1.00];
cfg.integrity.protocolVersion = 5;

%% Behavioral probe
cfg.behavior.referenceCount = 256;
cfg.behavior.madMultiplier = 3.0;
cfg.behavior.minimumSignals = 2;
cfg.behavior.minimumCosine = -0.25;
cfg.behavior.maximumNormRatio = 5.0;
cfg.behavior.enableLogitDrift = true;

%% Trust and aggregation
cfg.trust.initialAlpha = 1.0;
cfg.trust.initialBeta = 1.0;
cfg.trust.ema = 0.90;
cfg.trust.recoveryRate = 0.03;
cfg.trust.minimumWeight = 0.02;
cfg.aggregation.trimFraction = 0.20;
cfg.aggregation.flcopRobustBlend = 0.35;
cfg.aggregation.rfaIterations = 25;
cfg.aggregation.rfaTolerance = 1e-5;
cfg.aggregation.krumByzantineFraction = 0.20;

%% Compression and measured communication
cfg.compression.enabledForFLCOPPP = true;
cfg.compression.sparsity = 0.70;
cfg.compression.bits = 8;
cfg.communication.modelAfterGate = true;
cfg.communication.includeDownlink = true;
cfg.communication.includeEnrollment = true;
cfg.communication.modelScalarBytes = 4;
cfg.communication.indexBytes = 4;

%% Statistics
cfg.statistics.seeds = [17 43 91];
cfg.statistics.confidenceLevel = 0.95;
cfg.statistics.convergenceFractionOfCleanFedAvg = 0.90;

%% Outputs
cfg.outputs.saveMAT = true;
cfg.outputs.saveCSV = true;
cfg.outputs.saveExcel = true;
cfg.outputs.saveFigures = true;
cfg.outputs.figureResolution = 600;

%% Reviewer-facing ablations
cfg.ablation.enabled = false;
cfg.ablation.numRounds = 40;
cfg.ablation.numClients = [20 50 100 200];
cfg.ablation.challengeCounts = [2 4 8 16 32];
cfg.ablation.blockSizes = [1024 4096 16384];
cfg.ablation.nonIIDAlpha = [0.1 0.5 1.0];
cfg.ablation.maliciousFractions = [0.10 0.20 0.30];
cfg.ablation.sparsities = [0.0 0.5 0.7 0.9];
cfg.ablation.quantizationBits = [32 16 8 4];
cfg.ablation.componentMethods = [
    "FLCOPPP_NoPreGate"
    "FLCOPPP_NoBehavior"
    "FLCOPPP_NoTrust"
    "FLCOPPP_NoRobustCenter"
    "FLCOPPP_NoCompression"
    "FLCOPPP"
];

switch lower(profile)
    case "protocol_test"
        cfg.execution.runProtocolTestsBeforeSuite = true;
        cfg.fl.numRounds = 0;
        cfg.statistics.seeds = 17;

    case "smoke"
        cfg.fl.numClients = 10;
        cfg.fl.clientsPerRound = 4;
        cfg.fl.numRounds = 2;
        cfg.data.validationCount = 300;
        cfg.data.maxTrainSamplesPerClientRound = 128;
        cfg.behavior.referenceCount = 64;
        cfg.statistics.seeds = 17;
        cfg.suite.attacks = ["clean";"labelFlipAfterCommit";"backdoorBeforeCommit"];
        cfg.suite.methods = ["FedAvg";"TrimmedMean";"PreGateOnly";"FLCOPPP"];
        cfg.integrity.microbenchTrials = 30;
        cfg.integrity.baselineChallengeCounts = [2 8];
        cfg.integrity.baselineTamperFractions = [0.01 0.05];
        cfg.integrity.baselineDataFractions = 1.0;
        cfg.execution.checkpointEveryRounds = 1;
        cfg.fl.evalEvery = 1;
        cfg.attack.startRound = 1;

    case "paper_core"
        cfg.fl.numRounds = 60;
        cfg.data.maxTrainSamplesPerClientRound = 384;
        cfg.suite.attacks = [
            "clean"
            "labelFlipAfterCommit"
            "backdoorBeforeCommit"
            "modelReplacementBackdoor"
            "signFlip"
        ];
        cfg.suite.methods = [
            "FedAvg"
            "TrimmedMean"
            "Krum"
            "RFA"
            "FLTrust"
            "PreGateOnly"
            "FLCOPPP"
        ];
        cfg.integrity.microbenchTrials = 300;

    case "paper"
        % Exact manuscript-scale matrix: 5 scenarios x 7 baselines, 60 rounds,
        % 50 clients, 10 participants, 3 learning seeds. This is a NEW run;
        % never reuse older numerical results without verifying provenance.
        cfg.fl.numRounds = 60;
        cfg.suite.attacks = ["clean";"labelFlipAfterCommit"; ...
            "backdoorBeforeCommit";"modelReplacementBackdoor";"signFlip"];
        cfg.suite.methods = ["FedAvg";"TrimmedMean";"RFA";"Krum"; ...
            "FLTrust";"PreGateOnly";"FLCOPPP"];

    case "paper_with_ablations"
        cfg = flcoppp_config("paper");
        cfg.profile = profile;
        cfg.ablation.enabled = true;

    case "femnist_smoke"
        cfg.profile = "femnist_smoke";
        cfg.datasetName = "FEMNIST";
        cfg.fl.numClients = 4; cfg.fl.clientsPerRound = 2;
        cfg.fl.numRounds = 2; cfg.statistics.seeds = 17;
        cfg.data.validationCount = 20; cfg.behavior.referenceCount = 12;
        cfg.data.femnistMinUserSamples = 12;
        cfg.data.maxTrainSamplesPerClientRound = 40;
        cfg.suite.attacks = ["clean";"signFlip"];
        cfg.suite.methods = ["FedAvg";"FLCOPPP"];
        cfg.integrity.microbenchTrials = 10;
        cfg.integrity.baselineChallengeCounts = [2 8];
        cfg.integrity.baselineTamperFractions = 0.05;
        cfg.integrity.baselineDataFractions = 1;
        cfg.fl.evalEvery = 1; cfg.attack.startRound = 1;
        cfg.execution.checkpointEveryRounds = 1;

    otherwise
        error("Unknown profile: %s",profile);
end
if any(profile==["femnist_smoke"]) || cfg.datasetName=="FEMNIST"
    cfg = flcoppp_apply_femnist_config(cfg);
end
end
