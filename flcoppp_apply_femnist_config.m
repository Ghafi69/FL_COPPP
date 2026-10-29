function cfg = flcoppp_apply_femnist_config(cfg)
%FLCOPPP_APPLY_FEMNIST_CONFIG LEAF FEMNIST writer-partitioned configuration.
% Seed/method/round settings inherited from requested experiment profile.
cfg.datasetName = "FEMNIST";
cfg.data.numClasses = 62;
cfg.data.imageSize = [28 28 1];
cfg.data.channelMean = 0; cfg.data.channelStd = 1;
cfg.data.normalize = false; % raw LEAF pixels scaled to [0,1] by prepareInput
cfg.data.normalizationSource = "LEAF gray pixels scaled to [0,1], no train/test fitted statistics";
cfg.model.name = "CompactFEMNIST62ConvNet";
cfg.fl.minSamplesPerClient = 10;
cfg.fl.nonIIDAlpha = NaN; % writer-ID client partition, not Dirichlet
cfg.attack.backdoorPatchSize = 4;
cfg.attack.sourceLabel = 1; cfg.attack.targetLabel = 7;
end
