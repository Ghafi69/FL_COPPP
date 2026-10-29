function results = flcoppp_cifar10(cfg)
%FLCOPPP_CIFAR10 End-to-end FL-COP++ evaluation on official CIFAR-10 MAT files.
%
% This implementation is designed around the reviewer requests:
%   * actual non-IID federated training and convergence curves;
%   * label-flip, backdoor, model-replacement, and sign-flip attacks;
%   * FedAvg, coordinate median, trimmed mean, Krum, Multi-Krum, RFA,
%     FLTrust, pre-gate only, and full FL-COP++;
%   * CA, ASR, TPR/FPR, precision/F1, timing, and measured communication;
%   * deterministic cryptographic decisions with benign rejects reported
%     separately from transport/runtime failures;
%   * separate behavioral-detector FPR;
%   * structured tampering and exact hypergeometric-vs-approximate analysis;
%   * explicit dual-copy limitation: possession proofs do not prove that
%     committed data were actually used for local training.
%
% The code does not claim TEE-like execution integrity.

validateEnvironment(cfg);
cfg.execution.codeFile = [mfilename('fullpath') '.m'];
cfg = attachProtocolDigests(cfg);

protocolReport = struct('Passed',true);
if cfg.execution.runProtocolTestsBeforeSuite
    protocolReport = flcoppp_protocol("selftest",secureRandomBytes(32), ...
        min(cfg.integrity.blockSize,512),cfg.integrity.codeDigest,cfg.integrity.configDigest);
    if ~protocolReport.Passed
        error('Separated FL-COP++ protocol self-test failed.');
    end
end
cfg.execution.protocolSelfTestPassed=protocolReport.Passed;
if cfg.profile=="protocol_test"
    results=struct('outputDir',"",'ProtocolSelfTest',protocolReport);
    fprintf('Separated protocol tests: PASS\n');
    return;
end

writerIDs = [];
if string(cfg.datasetName)=="FEMNIST"
    datasetDir = string(cfg.paths.femnistRoot);
    [XTrain,YTrain,XTest,YTest,classNames,writerIDs,writerNames] = ...
        flcoppp_load_femnist(datasetDir,cfg);
else
    datasetDir = resolveCifarFolder(cfg.paths.cifarRoot);
    [XTrain, YTrain, XTest, YTest, classNames] = loadCIFAR10(datasetDir);
    writerNames = strings(0,1);
end

timestamp = char(datetime('now','Format','yyyyMMdd_HHmmss_SSS'));
outputDir = fullfile(cfg.paths.outputRoot, "run_" + string(timestamp) + "_" + cfg.profile + "_" + cfg.datasetName);
ensureDir(outputDir);
ensureDir(fullfile(outputDir,"tables"));
ensureDir(fullfile(outputDir,"figures"));
ensureDir(fullfile(outputDir,"checkpoints"));
ensureDir(fullfile(outputDir,"logs"));

saveConfiguration(cfg, outputDir);
writeSystemLedger(cfg, datasetDir, XTrain, YTrain, XTest, YTest, classNames, outputDir);
if ~isempty(writerNames)
    writetable(table((1:numel(writerNames))',writerNames(:), ...
        'VariableNames',{'ClientID','LEAFWriterID'}), ...
        fullfile(outputDir,'logs','femnist_writer_selection.csv'));
end
writeProtocolTestReport(protocolReport,outputDir);

if cfg.execution.verbose
    fprintf('\nFL-COP++ CIFAR-10 suite\n');
    fprintf('Dataset folder: %s\n', cifarDir);
    fprintf('Profile: %s | clients=%d | rounds=%d | seeds=%s\n', ...
        cfg.profile, cfg.fl.numClients, cfg.fl.numRounds, mat2str(cfg.statistics.seeds));
    fprintf(['Metric note: ASR=N/A for clean or untargeted attacks; ' ...
        'TPR=N/A when a round contains no active malicious client.\n']);
end

allRound = cell(0,1);
allTrust = cell(0,1);
allClient = cell(0,1);
runLedgerRows = struct([]);

for s = 1:numel(cfg.statistics.seeds)
    seed = cfg.statistics.seeds(s);
    rng(seed,'twister');

    if ~isempty(writerIDs)
        % Reference data are disjoint from all writer-client training pools.
        [refIdx,poolIdx] = flcoppp_writer_reference_split( ...
            writerIDs,cfg.data.validationCount,seed,cfg.fl.numClients, ...
            cfg.fl.minSamplesPerClient);
    else
        [refIdx,poolIdx] = stratifiedHoldout(YTrain,cfg.data.validationCount,seed);
    end
    XRef = XTrain(:,:,:,refIdx);
    YRef = YTrain(refIdx);
    assert(isempty(intersect(refIdx,poolIdx)), 'Reference leakage: overlapping samples');
    writetable(table(repmat(seed,numel(refIdx),1),refIdx(:),double(YRef(:)), ...
        'VariableNames',{'Seed','SourceTrainingIndex','Label'}), ...
        fullfile(outputDir,'logs',sprintf('reference_indices_seed_%d.csv',seed)));

    clientIndices = cell(cfg.fl.numClients,1);
    if ~isempty(writerIDs)
        for c=1:cfg.fl.numClients
            clientIndices{c}=poolIdx(writerIDs(poolIdx)==c);
        end
    else
        poolLabels=YTrain(poolIdx);
        localPartitions=dirichletPartition(poolLabels,cfg.fl.numClients, ...
            cfg.fl.nonIIDAlpha,cfg.fl.minSamplesPerClient,seed);
        for c=1:cfg.fl.numClients
            clientIndices{c}=poolIdx(localPartitions{c});
        end
    end
    assert(all(cellfun(@numel,clientIndices)>=cfg.fl.minSamplesPerClient), ...
        'Some selected clients have fewer samples than cfg.fl.minSamplesPerClient');

    schedule = makeClientSchedule(cfg.fl.numRounds, cfg.fl.numClients, ...
        cfg.fl.clientsPerRound, seed + 10000);
    maliciousMask = chooseMaliciousClients(cfg.fl.numClients, ...
        cfg.attack.maliciousFraction, seed + 20000);

    rng(seed + 30000,'twister');
    initialNet = buildCIFARNetwork(cfg);
    [initialVector, vectorMeta] = networkToVector(initialNet);

    if cfg.execution.verbose
        fprintf('\nSeed %d: building %d ECDH-derived client session keys...\n', ...
            seed, cfg.fl.numClients);
    end
    crypto = cryptoSelfTest(seed);
    [clientKeys, keyLedger] = makeClientSessionKeys(cfg.fl.numClients, seed, ...
        cfg.integrity.bindingDigest);

    for a = 1:numel(cfg.suite.attacks)
        attackName = string(cfg.suite.attacks(a));

        for m = 1:numel(cfg.suite.methods)
            methodName = string(cfg.suite.methods(m));
            runTag = sprintf('seed_%d__%s__%s', seed, sanitizeName(attackName), sanitizeName(methodName));
            checkpointFile = fullfile(outputDir,"checkpoints",runTag + ".mat");
            checkpointSignature = makeCheckpointSignature(cfg,seed,attackName,methodName);

            if cfg.execution.resume && isfile(checkpointFile)
                S = load(checkpointFile,'completed','checkpointSignature', ...
                    'roundTable','trustTable','clientTable','runLedger');
                compatibleCheckpoint = isfield(S,'checkpointSignature') && ...
                    strcmp(string(S.checkpointSignature),string(checkpointSignature));
                if compatibleCheckpoint && isfield(S,'completed') && S.completed
                    if cfg.execution.verbose
                        fprintf('Skipping completed run: %s\n', runTag);
                    end
                    allRound{end+1,1} = S.roundTable;
                    allTrust{end+1,1} = S.trustTable;
                    allClient{end+1,1} = S.clientTable;
                    runLedgerRows = appendStructRow(runLedgerRows,S.runLedger);
                    continue;
                elseif ~compatibleCheckpoint
                    warning('Ignoring incompatible checkpoint: %s',checkpointFile);
                end
            end

            if cfg.execution.verbose
                fprintf('\nRunning seed=%d | attack=%s | method=%s\n', ...
                    seed, attackName, methodName);
            end

            runStart = tic;
            [roundTable, trustTable, clientTable, runLedger] = runSingleExperiment( ...
                cfg, seed, attackName, methodName, XTrain, YTrain, XTest, YTest, ...
                XRef, YRef, clientIndices, schedule, maliciousMask, initialNet, ...
                initialVector, vectorMeta, clientKeys, keyLedger, checkpointFile, ...
                checkpointSignature);
            runLedger.TotalRunSeconds = toc(runStart);
            runLedger.CryptoSelfTestPassed = crypto.Passed;

            completed = true; %#ok<NASGU>
            save(checkpointFile,'completed','checkpointSignature','roundTable', ...
                'trustTable','clientTable','runLedger','-v7.3');

            allRound{end+1,1} = roundTable;
            allTrust{end+1,1} = trustTable;
            allClient{end+1,1} = clientTable;
            runLedgerRows = appendStructRow(runLedgerRows,runLedger);
        end
    end
end

RoundMetrics = vertcat(allRound{:});
TrustMetrics = vertcat(allTrust{:});
ClientDecisions = vertcat(allClient{:});
RunLedger = struct2table(runLedgerRows);

FinalSummary = summarizeFinalMetrics(RoundMetrics, ClientDecisions, cfg);
DetectionSummary = summarizeDetection(ClientDecisions, cfg);
DetectorAttribution = summarizeDetectorAttribution(ClientDecisions, cfg);
CryptoFailureAttribution = summarizeCryptoFailureReasons(ClientDecisions);
CommunicationSummary = summarizeCommunication(RoundMetrics);
RuntimeSummary = summarizeRuntime(RoundMetrics);
ComplexityTable = buildComplexityTable(cfg, initialVector, clientIndices);

% Reviewer-facing integrity experiments use one real CIFAR client byte stream.
sampleIdx = clientIndices{1};
sampleCount = min(numel(sampleIdx), 500);
sampleIdx = sampleIdx(1:sampleCount);
sampleBytes = serializeClientData(XTrain(:,:,:,sampleIdx), YTrain(sampleIdx));
[IntegrityMicrobenchmark, IntegrityBaselines, TheoryEmpirical] = ...
    runIntegrityMicrobenchmarks(sampleBytes, clientKeys{1}.SessionKey, cfg);

AblationResults = table();
if cfg.ablation.enabled
    AblationResults = runAblationSuite(cfg, XTrain, YTrain, XTest, YTest, ...
        XRef, YRef, outputDir);
end

exportAllTables(outputDir, cfg, RoundMetrics, TrustMetrics, ClientDecisions, ...
    RunLedger, FinalSummary, DetectionSummary, DetectorAttribution, CryptoFailureAttribution, CommunicationSummary, ...
    RuntimeSummary, ComplexityTable, IntegrityMicrobenchmark, ...
    IntegrityBaselines, TheoryEmpirical, AblationResults);

if cfg.outputs.saveFigures
    generateFigures(outputDir, cfg, RoundMetrics, TrustMetrics, FinalSummary, ...
        DetectionSummary, CommunicationSummary, RuntimeSummary, ...
        IntegrityMicrobenchmark, IntegrityBaselines, TheoryEmpirical, ...
        AblationResults);
end

if cfg.outputs.saveMAT
    save(fullfile(outputDir,'FLCOPPP_complete_results.mat'), ...
        'cfg','RoundMetrics','TrustMetrics','ClientDecisions','RunLedger', ...
        'FinalSummary','DetectionSummary','DetectorAttribution','CryptoFailureAttribution','CommunicationSummary','RuntimeSummary', ...
        'ComplexityTable','IntegrityMicrobenchmark','IntegrityBaselines', ...
        'TheoryEmpirical','AblationResults','-v7.3');
end

writeTextSummary(outputDir, cfg, FinalSummary, DetectionSummary, ...
    CommunicationSummary, RuntimeSummary, IntegrityMicrobenchmark);
flcoppp_export_reviewer_audit(outputDir,RoundMetrics,ClientDecisions,RunLedger,cfg);

results = struct();
results.outputDir = outputDir;
results.FinalSummary = FinalSummary;
results.DetectionSummary = DetectionSummary;
results.DetectorAttribution = DetectorAttribution;
results.CryptoFailureAttribution = CryptoFailureAttribution;
results.CommunicationSummary = CommunicationSummary;
results.RuntimeSummary = RuntimeSummary;
results.IntegrityMicrobenchmark = IntegrityMicrobenchmark;

fprintf('\nCompleted. Outputs saved to:\n%s\n', outputDir);
end

%% ========================================================================
function [roundTable, trustTable, clientTable, runLedger] = runSingleExperiment( ...
    cfg, seed, attackName, methodName, XTrain, YTrain, XTest, YTest, ...
    XRef, YRef, clientIndices, schedule, maliciousMask, initialNet, ...
    initialVector, vectorMeta, clientKeys, keyLedger, checkpointFile, ...
    checkpointSignature)

flags = methodFlags(methodName);
flags.useGate = flags.useGate && cfg.integrity.enabled;
flags.useCompression = flags.useCompression && cfg.compression.enabledForFLCOPPP;
numClients = cfg.fl.numClients;
numParams = numel(initialVector);
globalNet = buildCIFARNetwork(cfg);
globalNet = vectorToNetwork(globalNet,initialVector,vectorMeta);
globalVector = initialVector;

trust = cfg.trust.initialAlpha ./ (cfg.trust.initialAlpha + cfg.trust.initialBeta) ...
    * ones(numClients,1);
trustAlpha = cfg.trust.initialAlpha * ones(numClients,1);
trustBeta = cfg.trust.initialBeta * ones(numClients,1);

clientStates = cell(numClients,1);
serverRecords = cell(numClients,1);
enrollmentSeconds = zeros(numClients,1);
enrollmentBytes = zeros(numClients,1);

if flags.useGate
    for c = 1:numClients
        [Xc0,Yc0] = materializeClientData(XTrain,YTrain,clientIndices{c}, ...
            c, maliciousMask(c), attackName, 0, cfg, true);
        bytes0 = serializeClientData(Xc0,Yc0);
        t = tic;
        [clientStates{c},serverRecords{c},registrationBytes] = flcoppp_protocol( ...
            "enroll",bytes0,clientKeys{c}.SessionKey,cfg.integrity.blockSize, ...
            cfg.integrity.codeDigest,cfg.integrity.configDigest,c);
        [registrationSignature,bindingOK] = signAndVerifyRegistration( ...
            clientKeys{c},serverRecords{c}.RegistrationDigest);
        if ~bindingOK
            error('Enrollment signature verification failed for client %d.',c);
        end
        keyLedger(c).SignatureBytes=numel(registrationSignature);
        keyLedger(c).BindingVerified=bindingOK;
        enrollmentSeconds(c) = toc(t);
        enrollmentBytes(c) = registrationBytes + keyLedger(c).ClientPublicKeyBytes + ...
            keyLedger(c).ServerPublicKeyBytes + keyLedger(c).SigningPublicKeyBytes + ...
            keyLedger(c).SignatureBytes;
    end
end

roundRows = struct([]);
trustRows = struct([]);
clientRows = struct([]);
startRound = 1;

if cfg.execution.resume && isfile(checkpointFile)
    Sresume = load(checkpointFile);
    compatibleCheckpoint = isfield(Sresume,'checkpointSignature') && ...
        strcmp(string(Sresume.checkpointSignature),string(checkpointSignature));
    if compatibleCheckpoint && isfield(Sresume,'completed') && ~Sresume.completed && ...
            isfield(Sresume,'globalVector') && isfield(Sresume,'round')
        globalVector = Sresume.globalVector;
        globalNet = vectorToNetwork(globalNet,globalVector,vectorMeta);
        trust = Sresume.trust;
        trustAlpha = Sresume.trustAlpha;
        trustBeta = Sresume.trustBeta;
        startRound = Sresume.round + 1;
        if isfield(Sresume,'partialRoundTable') && ~isempty(Sresume.partialRoundTable)
            roundRows = table2struct(Sresume.partialRoundTable);
        end
        if isfield(Sresume,'partialTrustTable') && ~isempty(Sresume.partialTrustTable)
            trustRows = table2struct(Sresume.partialTrustTable);
        end
        if isfield(Sresume,'partialClientTable') && ~isempty(Sresume.partialClientTable)
            clientRows = table2struct(Sresume.partialClientTable);
        end
        if cfg.execution.verbose
            fprintf('  Resuming from round %d\n',startRound);
        end
    elseif ~compatibleCheckpoint
        warning('Stale checkpoint schema/code detected; restarting this run from round 1.');
    end
end

for round = startRound:cfg.fl.numRounds
    roundTimer = tic;
    selected = schedule(round,:);
    selected = selected(:)';
    nSelected = numel(selected);

    probeCounts = cfg.integrity.challengeCount*ones(1,nSelected);
    if flags.useGate && flags.useTrust && cfg.integrity.adaptiveProbeAllocation
        totalBudget = cfg.integrity.challengeCount*nSelected;
        probeCounts = allocateChallengeBudget(trust(selected),totalBudget, ...
            cfg.integrity.minimumChallengesPerClient);
    end

    updateMatrix = zeros(numParams,nSelected,'single');
    sampleCounts = zeros(1,nSelected);
    clientIds = zeros(1,nSelected);
    maliciousActive = false(1,nSelected);
    preGatePass = true(1,nSelected);
    secondaryGatePass = true(1,nSelected);
    behaviorFlag = false(1,nSelected);
    trained = false(1,nSelected);
    logitDrifts = nan(1,nSelected);
    compressionRatios = ones(1,nSelected);
    updateBytesByClient = zeros(1,nSelected);
    challengesByClient = zeros(1,nSelected);
    integrityBytesByClient = zeros(1,nSelected);
    challengeBytesByClient = zeros(1,nSelected);
    proofBytesByClient = zeros(1,nSelected);
    integritySecondsByClient = zeros(1,nSelected);
    serializationSecondsByClient = zeros(1,nSelected);
    challengeSecondsByClient = zeros(1,nSelected);
    proofGenerationSecondsByClient = zeros(1,nSelected);
    serverVerificationSecondsByClient = zeros(1,nSelected);
    challengeDigestByClient = strings(1,nSelected);
    challengeIndicesByClient = strings(1,nSelected);
    challengeNonceByClient = strings(1,nSelected);
    secondaryChallengesByClient = zeros(1,nSelected);
    verificationReasonByClient = repmat("not_applicable",1,nSelected);
    secondaryIntegritySecondsByClient = zeros(1,nSelected);
    currentBytesCache = cell(1,nSelected);
    localSecondsByClient = zeros(1,nSelected);

    downlinkBytes = 0;
    challengeBytes = 0;
    proofBytes = 0;
    integrityBytes = 0;
    updateBytes = 0;
    if round==1 && flags.useGate && cfg.communication.includeEnrollment
        enrollmentBytesRound = sum(enrollmentBytes);
    else
        enrollmentBytesRound = 0;
    end
    localTrainSeconds = 0;
    integritySeconds = 0;
    behaviorSeconds = 0;
    aggregationSeconds = 0;

    currentLR = cfg.fl.learnRate * cfg.fl.learnRateDecay^(round-1);
    globalRefProb = [];
    if flags.useBehavior
        globalRefProb = predictProbabilities(globalNet, ...
            XRef(:,:,:,1:min(end,cfg.behavior.referenceCount)), cfg);
    end

    for j = 1:nSelected
        c = selected(j);
        clientIds(j) = c;
        active = isAttackActive(attackName, maliciousMask(c), round, cfg);
        maliciousActive(j) = active;

        [Xc,Yc] = materializeClientData(XTrain,YTrain,clientIndices{c}, ...
            c, maliciousMask(c), attackName, round, cfg, false);
        sampleCounts(j) = numel(Yc);

        if flags.useGate
            tGateTotal = tic;

            tSerialize = tic;
            currentBytes = serializeClientData(Xc,Yc);
            serializationSecondsByClient(j) = toc(tSerialize);
            currentBytesCache{j} = currentBytes;

            qClient = min(probeCounts(j),double(serverRecords{c}.NumBlocks));
            challengesByClient(j) = qClient;

            tChallenge = tic;
            [challenge,serverRecords{c},cb] = flcoppp_protocol( ...
                "challenge",serverRecords{c},round,qClient);
            challengeSecondsByClient(j) = toc(tChallenge);
            challengeDigestByClient(j) = string(bytesToHex(challenge.Digest));
            challengeIndicesByClient(j) = strjoin(string(challenge.Indices(:)'),';');
            challengeNonceByClient(j) = string(bytesToHex(challenge.Nonce));

            tProof = tic;
            [packet,~] = flcoppp_protocol("prove",clientStates{c},currentBytes,challenge);
            proofGenerationSecondsByClient(j) = toc(tProof);

            tVerify = tic;
            [pass,serverRecords{c},verifyDiag] = flcoppp_protocol( ...
                "verify",serverRecords{c},challenge,packet);
            serverVerificationSecondsByClient(j) = toc(tVerify);
            verificationReasonByClient(j) = string(verifyDiag.FailureReason);

            integritySecondsByClient(j) = toc(tGateTotal);
            preGatePass(j) = pass;
            challengeBytesByClient(j) = cb;
            proofBytesByClient(j) = packet.WireBytes;
            integrityBytesByClient(j) = cb + packet.WireBytes;
            challengeBytes = challengeBytes + cb;
            proofBytes = proofBytes + packet.WireBytes;
            integrityBytes = integrityBytes + integrityBytesByClient(j);
            integritySeconds = integritySeconds + integritySecondsByClient(j);

            if ~pass && cfg.execution.verbose
                fprintf('    pre-gate reject client=%d reason=%s\n',c,verifyDiag.FailureReason);
            end
            if ~pass
                continue;
            end
        end

        if cfg.communication.includeDownlink
            if ~cfg.communication.modelAfterGate || preGatePass(j)
                downlinkBytes = downlinkBytes + 4*numParams;
            end
        end

        tLocal = tic;
        localNet = localTrain(globalNet, Xc, Yc, cfg, currentLR, ...
            seed + round*1000 + c);
        localSecondsByClient(j) = toc(tLocal);
        localTrainSeconds = localTrainSeconds + localSecondsByClient(j);

        [localVector,~] = networkToVector(localNet);
        update = localVector - globalVector;
        update = applyModelAttack(update, localVector, globalVector, vectorMeta, ...
            attackName, active, cfg);

        if flags.useCompression
            [receivedUpdate, bytesSent, compressionRatio] = ...
                compressUpdate(update, cfg.compression.sparsity, cfg.compression.bits);
        else
            receivedUpdate = update;
            bytesSent = 4*numParams;
            compressionRatio = 1.0;
        end

        updateMatrix(:,j) = receivedUpdate;
        updateBytesByClient(j) = bytesSent;
        compressionRatios(j) = compressionRatio;
        updateBytes = updateBytes + bytesSent;
        trained(j) = true;

        if flags.useBehavior && cfg.behavior.enableLogitDrift
            receivedNet = vectorToNetwork(globalNet, globalVector + receivedUpdate, vectorMeta);
            localProb = predictProbabilities(receivedNet, ...
                XRef(:,:,:,1:min(end,cfg.behavior.referenceCount)), cfg);
            logitDrifts(j) = mean(abs(localProb(:)-globalRefProb(:)));
        end
    end

    candidate = find(trained);
    behaviorStats = emptyBehaviorStats();

    if flags.useBehavior && ~isempty(candidate)
        tBehavior = tic;
        behaviorStats = behavioralProbe(updateMatrix(:,candidate), ...
            logitDrifts(candidate), cfg);
        behaviorFlag(candidate) = behaviorStats.Flag;
        behaviorSeconds = toc(tBehavior);

        % Secondary expanded challenge for clients flagged by the behavioral
        % probe. This can catch post-commitment tampering missed by the first
        % random sample, but it cannot validate initially poisoned data or
        % bind the update to the committed bytes.
        if flags.useGate
            suspicious = find(behaviorFlag & trained);
            for jj = suspicious
                tSecondary = tic;
                cid = clientIds(jj);
                q2 = min(double(serverRecords{cid}.NumBlocks), ...
                    max(challengesByClient(jj),cfg.integrity.challengeCount)* ...
                    cfg.integrity.secondaryChallengeMultiplier);
                [challenge2,serverRecords{cid},cb2] = flcoppp_protocol( ...
                    "challenge",serverRecords{cid},round,q2);
                [packet2,~] = flcoppp_protocol( ...
                    "prove",clientStates{cid},currentBytesCache{jj},challenge2);
                [pass2,serverRecords{cid},~] = flcoppp_protocol( ...
                    "verify",serverRecords{cid},challenge2,packet2);
                secondaryIntegritySecondsByClient(jj) = toc(tSecondary);
                secondaryChallengesByClient(jj) = numel(challenge2.Indices);
                secondaryGatePass(jj) = pass2;
                integritySecondsByClient(jj) = integritySecondsByClient(jj) + ...
                    secondaryIntegritySecondsByClient(jj);
                challengeBytesByClient(jj) = challengeBytesByClient(jj) + cb2;
                proofBytesByClient(jj) = proofBytesByClient(jj) + packet2.WireBytes;
                integrityBytesByClient(jj) = integrityBytesByClient(jj) + cb2 + packet2.WireBytes;
                challengeBytes = challengeBytes + cb2;
                proofBytes = proofBytes + packet2.WireBytes;
                integritySeconds = integritySeconds + secondaryIntegritySecondsByClient(jj);
                integrityBytes = integrityBytes + cb2 + packet2.WireBytes;
            end
        end
    end

    accepted = trained & ~behaviorFlag;
    rejected = ~preGatePass | behaviorFlag;

    if any(accepted)
        tAgg = tic;
        U = updateMatrix(:,accepted);
        W = sampleCounts(accepted);
        IDs = clientIds(accepted);
        rootUpdate = [];
        if methodName == "FLTrust"
            rootNet = localTrain(globalNet, ...
                XRef(:,:,:,1:min(end,cfg.behavior.referenceCount)), ...
                YRef(1:min(end,cfg.behavior.referenceCount)), cfg, currentLR, ...
                seed + round*50000);
            rootVector = networkToVector(rootNet);
            rootUpdate = rootVector - globalVector;
        end

        aggregate = aggregateUpdates(methodName, U, W, IDs, trust, ...
            rootUpdate, cfg);
        globalVector = globalVector + aggregate;
        globalNet = vectorToNetwork(globalNet, globalVector, vectorMeta);
        aggregationSeconds = toc(tAgg);
    end

    [trust,trustAlpha,trustBeta] = updateTrust(trust,trustAlpha,trustBeta, ...
        selected, preGatePass, behaviorFlag, trained, cfg);

    if mod(round,cfg.fl.evalEvery)==0 || round==1 || round==cfg.fl.numRounds
        [CA,testLoss] = evaluateCleanAccuracy(globalNet,XTest,YTest,cfg);
        ASR = evaluateASR(globalNet,XTest,YTest,attackName,cfg);
    else
        CA = NaN;
        ASR = NaN;
        testLoss = NaN;
    end

    activeMask = maliciousActive;
    benignMask = ~maliciousActive;
    TP = sum(rejected & activeMask);
    FN = sum(~rejected & activeMask);
    FP = sum(rejected & benignMask);
    TN = sum(~rejected & benignMask);
    [TPR,FPR,precision,F1] = detectionMetrics(TP,FN,FP,TN);

    totalBytes = downlinkBytes + updateBytes + integrityBytes + enrollmentBytesRound;
    assert(integrityBytes==challengeBytes+proofBytes, ...
        'Integrity accounting mismatch: challenge/proof totals differ');
    assert(totalBytes>=0 && isfinite(totalBytes), 'Invalid communication total');
    roundSeconds = toc(roundTimer);

    rr = struct();
    rr.Dataset = string(cfg.datasetName);
    rr.Seed = seed;
    rr.Method = methodName;
    rr.Attack = attackName;
    rr.Round = round;
    rr.CleanAccuracy = CA;
    rr.AttackSuccessRate = ASR;
    rr.TestLoss = testLoss;
    rr.TPR = TPR;
    rr.FPR = FPR;
    rr.Precision = precision;
    rr.F1 = F1;
    rr.TP = TP; rr.FN = FN; rr.FP = FP; rr.TN = TN;
    rr.SelectedClients = nSelected;
    rr.TrainedClients = sum(trained);
    rr.AcceptedClients = sum(accepted);
    rr.QuarantinedClients = sum(rejected);
    rr.ActiveMaliciousClients = sum(activeMask);
    rr.RoundSeconds = roundSeconds;
    if round==1, rr.EnrollmentSeconds=sum(enrollmentSeconds); else, rr.EnrollmentSeconds=0; end
    rr.IntegritySeconds = integritySeconds;
    rr.LocalTrainingSeconds = localTrainSeconds;
    rr.BehaviorSeconds = behaviorSeconds;
    rr.AggregationSeconds = aggregationSeconds;
    rr.DownlinkBytes = downlinkBytes;
    rr.UpdateBytes = updateBytes;
    rr.ChallengeBytes = challengeBytes;
    rr.ProofBytes = proofBytes;
    rr.IntegrityBytes = integrityBytes;
    rr.EnrollmentBytes = enrollmentBytesRound;
    rr.TotalBytes = totalBytes;
    rr.IntegrityOverheadPercent = safeDivide(100*(integrityBytes+enrollmentBytesRound),totalBytes);
    rr.MeanCompressionRatio = mean(compressionRatios(trained),'omitnan');
    rr.MeanTrust = mean(trust);
    rr.MeanBenignTrust = mean(trust(~maliciousMask));
    if any(maliciousMask)
        rr.MeanMaliciousTrust = mean(trust(maliciousMask));
    else
        rr.MeanMaliciousTrust = NaN;
    end
    roundRows = appendStructRow(roundRows,rr);

    tr = struct();
    tr.Dataset = string(cfg.datasetName);
    tr.Seed = seed;
    tr.Method = methodName;
    tr.Attack = attackName;
    tr.Round = round;
    tr.MeanBenignTrust = rr.MeanBenignTrust;
    tr.MeanMaliciousTrust = rr.MeanMaliciousTrust;
    tr.TrustGap = tr.MeanBenignTrust - tr.MeanMaliciousTrust;
    trustRows = appendStructRow(trustRows,tr);

    for j = 1:nSelected
        cr = struct();
        cr.Dataset = string(cfg.datasetName);
        cr.Seed = seed;
        cr.Method = methodName;
        cr.Attack = attackName;
        cr.Round = round;
        cr.ClientID = clientIds(j);
        cr.IsMalicious = maliciousMask(clientIds(j));
        cr.AttackActive = maliciousActive(j);
        cr.PreGatePass = preGatePass(j);
        cr.SecondaryGatePass = secondaryGatePass(j);
        cr.BehaviorFlag = behaviorFlag(j);
        cr.Trained = trained(j);
        cr.IsAssignedAdversary = maliciousMask(clientIds(j));
        cr.Rejected = rejected(j);
        cr.TrustAfterRound = trust(clientIds(j));
        cr.LogitDrift = logitDrifts(j);
        cr.UpdateBytes = updateBytesByClient(j);
        cr.ChallengesUsed = challengesByClient(j); % PRIMARY challenge count only
        cr.SecondaryChallengesUsed = secondaryChallengesByClient(j);
        cr.TotalChallengesUsed = challengesByClient(j)+secondaryChallengesByClient(j);
        cr.ChallengeIndices = challengeIndicesByClient(j);
        cr.ChallengeNonceHex = challengeNonceByClient(j);
        cr.ChallengeBytes = challengeBytesByClient(j);
        cr.ProofBytes = proofBytesByClient(j);
        cr.IntegrityBytes = integrityBytesByClient(j);
        cr.IntegritySeconds = integritySecondsByClient(j);
        cr.SerializationSeconds = serializationSecondsByClient(j);
        cr.ChallengeGenerationSeconds = challengeSecondsByClient(j);
        cr.ProofGenerationSeconds = proofGenerationSecondsByClient(j);
        cr.ServerVerificationSeconds = serverVerificationSecondsByClient(j);
        cr.ChallengeDigestHex = challengeDigestByClient(j);
        cr.PreGateFailureReason = verificationReasonByClient(j);
        cr.SecondaryIntegritySeconds = secondaryIntegritySecondsByClient(j);
        cr.LocalTrainingSeconds = localSecondsByClient(j);
        clientRows = appendStructRow(clientRows,cr);
    end

    if cfg.execution.verbose
        fprintf(['  r=%3d CA=%6.2f%% ASR=%s accepted=%d/%d activeMal=%d ' ...
            'TPR=%s FPR=%s bytes=%.2f MB time=%.2fs\n'], ...
            round,100*CA,fmtPercentOrNA(ASR),sum(accepted),nSelected, ...
            sum(activeMask),fmtRate(TPR),fmtRate(FPR), ...
            totalBytes/1024^2,roundSeconds);
    end

    if cfg.execution.saveCheckpoints && ...
            (mod(round,cfg.execution.checkpointEveryRounds)==0 || round==cfg.fl.numRounds)
        partialRoundTable = struct2table(roundRows); %#ok<NASGU>
        partialTrustTable = struct2table(trustRows); %#ok<NASGU>
        partialClientTable = struct2table(clientRows); %#ok<NASGU>
        completed = false; %#ok<NASGU>
        save(checkpointFile,'completed','checkpointSignature','partialRoundTable', ...
            'partialTrustTable','partialClientTable','globalVector','trust', ...
            'trustAlpha','trustBeta','round','-v7.3');
    end
end

roundTable = struct2table(roundRows);
trustTable = struct2table(trustRows);
clientTable = struct2table(clientRows);

runLedger = struct();
runLedger.Dataset = string(cfg.datasetName);
runLedger.Seed = seed;
runLedger.Method = methodName;
runLedger.Attack = attackName;
runLedger.Clients = cfg.fl.numClients;
runLedger.ClientsPerRound = cfg.fl.clientsPerRound;
runLedger.Rounds = cfg.fl.numRounds;
runLedger.LocalEpochs = cfg.fl.localEpochs;
runLedger.NonIIDAlpha = cfg.fl.nonIIDAlpha;
runLedger.MaliciousFraction = cfg.attack.maliciousFraction;
runLedger.BlockSize = cfg.integrity.blockSize;
runLedger.Challenges = cfg.integrity.challengeCount;
runLedger.AdaptiveProbeAllocation = cfg.integrity.adaptiveProbeAllocation;
runLedger.ProbePolicy = string(cfg.integrity.probePolicy);
runLedger.SecondaryChallengeMultiplier = cfg.integrity.secondaryChallengeMultiplier;
runLedger.BehaviorMADMultiplier = cfg.behavior.madMultiplier;
runLedger.BehaviorMinimumSignals = cfg.behavior.minimumSignals;
runLedger.TrustEMA = cfg.trust.ema;
runLedger.TrustRecoveryRate = cfg.trust.recoveryRate;
runLedger.TrimFraction = cfg.aggregation.trimFraction;
runLedger.RobustBlend = cfg.aggregation.flcopRobustBlend;
runLedger.ReferenceSetAssumption = string(cfg.data.referenceSetAssumption);
runLedger.ProtocolVersion = cfg.integrity.protocolVersion;
runLedger.ServerReferenceCount = cfg.data.validationCount;
runLedger.BehaviorReferenceCount = cfg.behavior.referenceCount;
runLedger.NormalizationSource = string(cfg.data.normalizationSource);
runLedger.TargetLayerNames = strjoin(string(cfg.attack.targetLayerNames),',');
runLedger.Sparsity = double(flags.useCompression)*cfg.compression.sparsity;
if flags.useCompression
    runLedger.QuantizationBits = cfg.compression.bits;
else
    runLedger.QuantizationBits = 32;
end
runLedger.NumParameters = numParams;
runLedger.EnrollmentSeconds = sum(enrollmentSeconds);
runLedger.EnrollmentBytes = sum(enrollmentBytes);
runLedger.TotalRunSeconds = NaN;
runLedger.CryptoSelfTestPassed = false;
runLedger.SeparatedProtocolSelfTestPassed = cfg.execution.protocolSelfTestPassed;
end

%% ========================================================================
function flags = methodFlags(methodName)
methodName = string(methodName);
isFLCOP = startsWith(methodName,"FLCOPPP");
flags.useGate = methodName=="PreGateOnly" || ...
    (isFLCOP && methodName~="FLCOPPP_NoPreGate");
flags.useBehavior = isFLCOP && methodName~="FLCOPPP_NoBehavior";
flags.useCompression = isFLCOP && methodName~="FLCOPPP_NoCompression";
flags.useTrust = isFLCOP && methodName~="FLCOPPP_NoTrust";
end

function active = isAttackActive(attackName,isMalicious,round,cfg)
if ~isMalicious || attackName=="clean"
    active = false;
    return;
end
if any(attackName==["backdoorBeforeCommit","modelReplacementBackdoor"])
    active = true;
else
    active = round >= cfg.attack.startRound;
end
end

function [Xc,Yc] = materializeClientData(XTrain,YTrain,idx,clientID,isMalicious, ...
    attackName,roundIndex,cfg,forCommit)
Xc = XTrain(:,:,:,idx);
Yc = YTrain(idx);

if ~isMalicious || attackName=="clean"
    return;
end

beforeCommitPoison = any(attackName==["backdoorBeforeCommit","modelReplacementBackdoor"]);
active = isAttackActive(attackName,isMalicious,max(roundIndex,1),cfg);

if forCommit
    applyPoison = beforeCommitPoison;
else
    applyPoison = active && any(attackName==["labelFlipAfterCommit", ...
        "backdoorBeforeCommit","modelReplacementBackdoor"]);
end

if ~applyPoison
    return;
end

n = numel(Yc);
k = max(1,round(cfg.attack.poisonFraction*n));
stream = RandStream('mt19937ar','Seed',100000 + clientID*97 + double(sum(char(attackName))));
order = randperm(stream,n,k);

if attackName=="labelFlipAfterCommit"
    source = uint8(cfg.attack.sourceLabel);
    target = uint8(cfg.attack.targetLabel);
    sourceIdx = find(Yc==source);
    if isempty(sourceIdx)
        chosen = order;
    else
        k2 = min(k,numel(sourceIdx));
        chosen = sourceIdx(randperm(stream,numel(sourceIdx),k2));
    end
    Yc(chosen) = target;
else
    target = uint8(cfg.attack.targetLabel);
    patch = cfg.attack.backdoorPatchSize;
    r = (size(Xc,1)-patch+1):size(Xc,1);
    c = (size(Xc,2)-patch+1):size(Xc,2);
    Xc(r,c,:,order) = cfg.attack.backdoorValue;
    Yc(order) = target;
end
end

function update = applyModelAttack(update,localVector,globalVector,vectorMeta, ...
    attackName,active,cfg)
if ~active
    return;
end
switch attackName
    case "modelReplacementBackdoor"
        scale = cfg.attack.modelReplacementScale;
        if scale<=0
            % Classical model-replacement scaling approximately cancels the
            % averaging dilution introduced by participating clients.
            scale = cfg.fl.clientsPerRound;
        end
        update = scale * update;
    case "signFlip"
        update = -cfg.attack.signFlipScale * update;
    case "targetedLayerReplacement"
        scale = cfg.attack.modelReplacementScale;
        if scale<=0, scale = cfg.fl.clientsPerRound; end
        targetNames = string(cfg.attack.targetLayerNames);
        offset = 0;
        selectedAny = false;
        for r = 1:numel(vectorMeta.counts)
            n = vectorMeta.counts(r);
            idx = offset + (1:n);
            if any(vectorMeta.layers(r)==targetNames)
                update(idx) = scale * ...
                    (localVector(idx)-globalVector(idx));
                selectedAny = true;
            end
            offset = offset+n;
        end
        if ~selectedAny
            error('No learnable layer matched cfg.attack.targetLayerNames.');
        end
    otherwise
        % Data attacks modify the local examples rather than the update.
end
end

%% ========================================================================
function net = buildCIFARNetwork(cfg)
layers = [
    imageInputLayer(cfg.data.imageSize,'Normalization','none','Name','input')
    convolution2dLayer(3,32,'Padding','same','WeightsInitializer','he','Name','conv1')
    reluLayer('Name','relu1')
    convolution2dLayer(3,32,'Padding','same','WeightsInitializer','he','Name','conv2')
    reluLayer('Name','relu2')
    maxPooling2dLayer(2,'Stride',2,'Name','pool1')
    dropoutLayer(cfg.model.dropout,'Name','drop1')
    convolution2dLayer(3,64,'Padding','same','WeightsInitializer','he','Name','conv3')
    reluLayer('Name','relu3')
    convolution2dLayer(3,64,'Padding','same','WeightsInitializer','he','Name','conv4')
    reluLayer('Name','relu4')
    maxPooling2dLayer(2,'Stride',2,'Name','pool2')
    dropoutLayer(cfg.model.dropout,'Name','drop2')
    convolution2dLayer(3,128,'Padding','same','WeightsInitializer','he','Name','conv5')
    reluLayer('Name','relu5')
    globalAveragePooling2dLayer('Name','gap')
    fullyConnectedLayer(cfg.data.numClasses,'WeightsInitializer','he','Name','fc')
    softmaxLayer('Name','softmax')
];
net = dlnetwork(layers);
end

function net = localTrain(net,X,Y,cfg,learnRate,seed)
rng(seed,'twister');
n = numel(Y);
maxLocal = cfg.data.maxTrainSamplesPerClientRound;
if isfinite(maxLocal) && n>maxLocal
    keep = randperm(n,maxLocal);
    X = X(:,:,:,keep);
    Y = Y(keep);
    n = numel(Y);
end
mb = min(cfg.fl.miniBatchSize,n);

for epoch = 1:cfg.fl.localEpochs
    order = randperm(n);
    for startIdx = 1:mb:n
        batchIdx = order(startIdx:min(startIdx+mb-1,n));
        dlX = prepareInput(X(:,:,:,batchIdx),cfg);
        dlT = oneHotTargets(Y(batchIdx),cfg.data.numClasses,cfg);
        [loss,grads] = dlfeval(@modelGradients,net,dlX,dlT); %#ok<ASGLU>
        grads = clipGradientTable(grads,cfg.fl.gradientClip);
        net = dlupdate(@(p,g) p - learnRate*g,net,grads);
    end
end
end

function [loss,grads] = modelGradients(net,dlX,dlT)
pred = forward(net,dlX);
loss = crossentropy(pred,dlT);
grads = dlgradient(loss,net.Learnables);
end

function grads = clipGradientTable(grads,clipValue)
sumSq = 0;
for i = 1:height(grads)
    g = extractdata(grads.Value{i});
    if isa(g,'gpuArray'), g = gather(g); end
    sumSq = sumSq + sum(double(g(:)).^2);
end
normVal = sqrt(sumSq);
scale = min(1,clipValue/(normVal+eps));
grads = dlupdate(@(g) g*scale,grads);
end

function dlX = prepareInput(X,cfg)
X = single(X)/255;
if cfg.data.normalize
    for c = 1:size(X,3)
        X(:,:,c,:) = (X(:,:,c,:) - cfg.data.channelMean(c)) / cfg.data.channelStd(c);
    end
end
if shouldUseGPU(cfg)
    X = gpuArray(X);
end
dlX = dlarray(X,'SSCB');
end

function dlT = oneHotTargets(Y,numClasses,cfg)
Y = double(Y(:))';
B = numel(Y);
T = zeros(numClasses,B,'single');
ind = sub2ind([numClasses,B],Y+1,1:B);
T(ind) = 1;
if shouldUseGPU(cfg)
    T = gpuArray(T);
end
dlT = dlarray(T,'CB');
end

function [vec,meta] = networkToVector(net)
L = net.Learnables;
meta = struct();
meta.sizes = cell(height(L),1);
meta.counts = zeros(height(L),1);
meta.layers = string(L.Layer);
meta.parameters = string(L.Parameter);
parts = cell(height(L),1);
for i = 1:height(L)
    p = extractdata(L.Value{i});
    if isa(p,'gpuArray'), p = gather(p); end
    p = single(p);
    meta.sizes{i} = size(p);
    meta.counts(i) = numel(p);
    parts{i} = p(:);
end
vec = vertcat(parts{:});
end

function net = vectorToNetwork(net,newVector,meta)
current = networkToVector(net);
deltaVector = single(newVector-current);
deltaTable = net.Learnables;
offset = 0;
for i = 1:height(deltaTable)
    n = meta.counts(i);
    d = reshape(deltaVector(offset+1:offset+n),meta.sizes{i});
    existing = extractdata(deltaTable.Value{i});
    if isa(existing,'gpuArray')
        d = gpuArray(d);
    end
    deltaTable.Value{i} = dlarray(d);
    offset = offset+n;
end
net = dlupdate(@(p,d) p+d,net,deltaTable);
end

function probs = predictProbabilities(net,X,cfg)
n = size(X,4);
mb = max(64,cfg.fl.miniBatchSize);
probs = zeros(cfg.data.numClasses,n,'single');
for s = 1:mb:n
    idx = s:min(s+mb-1,n);
    dlX = prepareInput(X(:,:,:,idx),cfg);
    y = predict(net,dlX);
    y = extractdata(y);
    if isa(y,'gpuArray'), y = gather(y); end
    probs(:,idx) = single(y);
end
end

function [acc,loss] = evaluateCleanAccuracy(net,X,Y,cfg)
p = predictProbabilities(net,X,cfg);
[~,pred] = max(p,[],1);
pred = uint8(pred-1);
acc = mean(pred(:)==Y(:));
trueIdx = sub2ind(size(p),double(Y(:)')+1,1:numel(Y));
loss = -mean(log(max(double(p(trueIdx)),1e-12)));
end

function asr = evaluateASR(net,X,Y,attackName,cfg)
target = uint8(cfg.attack.targetLabel);
source = uint8(cfg.attack.sourceLabel);

switch attackName
    case {"backdoorBeforeCommit","modelReplacementBackdoor"}
        keep = Y~=target;
        Xt = X(:,:,:,keep);
        patch = cfg.attack.backdoorPatchSize;
        r = (size(Xt,1)-patch+1):size(Xt,1);
        c = (size(Xt,2)-patch+1):size(Xt,2);
        Xt(r,c,:,:) = cfg.attack.backdoorValue;
        p = predictProbabilities(net,Xt,cfg);
        [~,pred] = max(p,[],1);
        asr = mean(uint8(pred-1)==target);

    case "labelFlipAfterCommit"
        keep = Y==source;
        if ~any(keep)
            asr = NaN;
            return;
        end
        p = predictProbabilities(net,X(:,:,:,keep),cfg);
        [~,pred] = max(p,[],1);
        asr = mean(uint8(pred-1)==target);

    otherwise
        asr = NaN;
end
end

%% ========================================================================
function aggregate = aggregateUpdates(methodName,U,sampleCounts,clientIDs,trust, ...
    rootUpdate,cfg)
methodName = string(methodName);
if size(U,2)==1
    aggregate = U(:,1);
    return;
end

switch methodName
    case {"FedAvg","PreGateOnly"}
        aggregate = weightedMean(U,sampleCounts);

    case "CoordinateMedian"
        aggregate = median(U,2);

    case "TrimmedMean"
        aggregate = coordinateTrimmedMean(U,cfg.aggregation.trimFraction);

    case "Krum"
        aggregate = krumAggregate(U,cfg,false);

    case "MultiKrum"
        aggregate = krumAggregate(U,cfg,true);

    case "RFA"
        aggregate = rfaAggregate(U,sampleCounts,cfg);

    case "FLTrust"
        aggregate = flTrustAggregate(U,sampleCounts,rootUpdate);

    case {"FLCOPPP","FLCOPPP_NoBehavior","FLCOPPP_NoCompression", ...
            "FLCOPPP_NoTrust","FLCOPPP_NoPreGate","FLCOPPP_NoRobustCenter"}
        if methodName=="FLCOPPP_NoRobustCenter"
            if methodName=="FLCOPPP_NoTrust"
                weights=double(sampleCounts);
            else
                weights=double(sampleCounts).*double(trust(clientIDs)');
            end
            aggregate=weightedMean(U,weights); % no trimmed center in weights or blend
            return;
        end
        center = coordinateTrimmedMean(U,cfg.aggregation.trimFraction);
        cosScore = zeros(1,size(U,2));
        for i = 1:size(U,2)
            cosScore(i) = max(0,cosineSimilarity(U(:,i),center));
        end
        if methodName=="FLCOPPP_NoTrust"
            clientTrust = ones(1,numel(clientIDs));
        else
            clientTrust = trust(clientIDs)';
        end
        w = double(sampleCounts).*double(clientTrust).*max(cosScore,cfg.trust.minimumWeight);
        if sum(w)<=0
            weighted = center;
        else
            weighted = U*(single(w(:)/sum(w)));
        end
        beta = cfg.aggregation.flcopRobustBlend;
        aggregate = (1-beta)*weighted + beta*center;

    otherwise
        error("Unknown aggregation method: %s",methodName);
end
end

function v = weightedMean(U,w)
w = double(w(:));
if sum(w)<=0, w = ones(size(w)); end
w = w/sum(w);
v = U*single(w);
end

function v = coordinateTrimmedMean(U,trimFraction)
n = size(U,2);
trim = floor(trimFraction*n);
if 2*trim>=n
    v = median(U,2);
    return;
end
S = sort(U,2);
v = mean(S(:,trim+1:n-trim),2);
end

function v = krumAggregate(U,cfg,multi)
n = size(U,2);
f = floor(cfg.aggregation.krumByzantineFraction*n);
if n <= 2*f+2
    v = median(U,2);
    return;
end
D = zeros(n,n);
for i = 1:n
    for j = i+1:n
        d = double(U(:,i)-U(:,j));
        D(i,j) = sum(d.^2);
        D(j,i) = D(i,j);
    end
end
neighborCount = n-f-2;
scores = zeros(n,1);
for i = 1:n
    d = sort(D(i,[1:i-1,i+1:n]));
    scores(i) = sum(d(1:neighborCount));
end
[~,order] = sort(scores,'ascend');
if multi
    keep = max(1,n-f-2);
    v = mean(U(:,order(1:keep)),2);
else
    v = U(:,order(1));
end
end

function v = rfaAggregate(U,w,cfg)
v = weightedMean(U,w);
w = double(w(:));
w = w/sum(w);
for it = 1:cfg.aggregation.rfaIterations
    distances = zeros(size(U,2),1);
    for j = 1:size(U,2)
        distances(j) = norm(double(U(:,j)-v))+1e-12;
    end
    ww = w./distances;
    ww = ww/sum(ww);
    vNew = U*single(ww);
    if norm(double(vNew-v)) <= cfg.aggregation.rfaTolerance*(norm(double(v))+1e-12)
        v = vNew;
        break;
    end
    v = vNew;
end
end

function v = flTrustAggregate(U,w,rootUpdate)
if isempty(rootUpdate) || norm(double(rootUpdate))<eps
    v = weightedMean(U,w);
    return;
end
rootNorm = norm(double(rootUpdate));
trustScore = zeros(1,size(U,2));
scaled = zeros(size(U),'single');
for j = 1:size(U,2)
    uj = U(:,j);
    ujNorm = norm(double(uj))+eps;
    trustScore(j) = max(0,double(dot(uj,rootUpdate))/(ujNorm*rootNorm+eps));
    scaled(:,j) = uj*single(rootNorm/ujNorm);
end
weights = double(w(:)').*trustScore;
if sum(weights)<=0
    v = rootUpdate;
else
    v = scaled*single(weights(:)/sum(weights));
end
end

function c = cosineSimilarity(a,b)
c = double(dot(a,b))/(norm(double(a))*norm(double(b))+eps);
end

%% ========================================================================
function stats = behavioralProbe(U,logitDrift,cfg)
n = size(U,2);
center = median(U,2);
norms = zeros(1,n);
cosines = zeros(1,n);
for j = 1:n
    norms(j) = norm(double(U(:,j)));
    cosines(j) = cosineSimilarity(U(:,j),center);
end

cosMed = median(cosines);
cosScale = robustScale(cosines);
cosThreshold = max(cfg.behavior.minimumCosine, ...
    cosMed-cfg.behavior.madMultiplier*cosScale);

normMed = median(norms);
normScale = robustScale(norms);
normThreshold = max(cfg.behavior.maximumNormRatio*normMed, ...
    normMed+cfg.behavior.madMultiplier*normScale);

if all(isnan(logitDrift))
    driftThreshold = Inf;
else
    d = logitDrift(~isnan(logitDrift));
    driftThreshold = median(d)+cfg.behavior.madMultiplier*robustScale(d);
end

cosBad = cosines<cosThreshold;
normBad = norms>normThreshold;
driftBad = false(1,n);
if cfg.behavior.enableLogitDrift
    driftBad = logitDrift>driftThreshold;
end
signalCount = double(cosBad)+double(normBad)+double(driftBad);
extremeNorm = norms > 2*normThreshold;
flag = signalCount>=cfg.behavior.minimumSignals | extremeNorm;

stats = struct();
stats.Flag = flag;
stats.SignalCount = signalCount;
stats.CosineBad = cosBad;
stats.NormBad = normBad;
stats.DriftBad = driftBad;
stats.Cosine = cosines;
stats.Norm = norms;
stats.LogitDrift = logitDrift;
stats.CosineThreshold = cosThreshold;
stats.NormThreshold = normThreshold;
stats.DriftThreshold = driftThreshold;
end

function s = robustScale(x)
x = double(x(:));
x = x(~isnan(x));
if isempty(x)
    s = 0;
    return;
end
m = median(x);
s = 1.4826*median(abs(x-m));
if s<1e-12
    s = std(x);
end
if s<1e-12
    s = 1e-12;
end
end

function s = emptyBehaviorStats()
s = struct('Flag',false(1,0),'SignalCount',[],'CosineBad',[],'NormBad',[], ...
    'DriftBad',[],'Cosine',[],'Norm',[],'LogitDrift',[], ...
    'CosineThreshold',NaN,'NormThreshold',NaN,'DriftThreshold',NaN);
end

function [trust,a,b] = updateTrust(trust,a,b,selected,prePass,behaviorFlag,trained,cfg)
for j = 1:numel(selected)
    c = selected(j);
    if ~prePass(j)
        evidence = 0.0;
    elseif behaviorFlag(j)
        evidence = 0.25;
    elseif trained(j)
        evidence = 1.0;
    else
        evidence = 0.5;
    end
    a(c) = a(c)+evidence;
    b(c) = b(c)+(1-evidence);
    posterior = a(c)/(a(c)+b(c));
    trust(c) = cfg.trust.ema*trust(c)+(1-cfg.trust.ema)*posterior;
    if evidence==1
        trust(c) = trust(c)+cfg.trust.recoveryRate*(1-trust(c));
    end
    trust(c) = min(max(trust(c),0),1);
end
end

function [TPR,FPR,precision,F1] = detectionMetrics(TP,FN,FP,TN)
TPR = safeDivide(TP,TP+FN);
FPR = safeDivide(FP,FP+TN);
precision = safeDivide(TP,TP+FP);
if isnan(precision) || isnan(TPR) || precision+TPR==0
    F1 = NaN;
else
    F1 = 2*precision*TPR/(precision+TPR);
end
end

%% ========================================================================
function q = allocateChallengeBudget(clientTrust,totalBudget,minPerClient)
% Risk-proportional allocation under a fixed per-round challenge budget.
% sqrt-risk prevents one low-trust client from consuming the entire budget.
n=numel(clientTrust);
minPerClient=min(minPerClient,floor(totalBudget/n));
q=minPerClient*ones(1,n);
remaining=totalBudget-sum(q);
if remaining<=0, return; end
risk=sqrt(max(1-double(clientTrust(:)'),1e-3));
weights=risk/sum(risk);
raw=remaining*weights;
extra=floor(raw);
q=q+extra;
left=totalBudget-sum(q);
[~,order]=sort(raw-extra,'descend');
if left>0
    q(order(1:left))=q(order(1:left))+1;
end
end

%% ========================================================================
function [decompressed,bytes,ratio] = compressUpdate(update,sparsity,bits)
n = numel(update);
keep = max(1,round((1-sparsity)*n));

if keep>=n && bits>=32
    decompressed = single(update);
    bytes = 4*n;
    ratio = 1;
    return;
end

[~,order] = sort(abs(update),'descend');
idx = order(1:keep);
vals = single(update(idx));

if bits>=32
    qvals = vals;
    valueBytes = 4*keep;
else
    maxAbs = max(abs(vals));
    if maxAbs==0
        qvals = zeros(size(vals),'single');
    else
        levels = 2^(bits-1)-1;
        q = round(double(vals)/double(maxAbs)*levels);
        q = max(-levels,min(levels,q));
        qvals = single(q/levels*double(maxAbs));
    end
    valueBytes = ceil(bits*keep/8);
end

decompressed = zeros(n,1,'single');
decompressed(idx) = qvals;
indexBytes = 4*keep;
metadataBytes = 16; % count, scale/range, bit width, sparse length
bytes = indexBytes + valueBytes + metadataBytes;
ratio = bytes/(4*n);
end

%% ========================================================================
function [clientKeys,ledger] = makeClientSessionKeys(numClients,seed,bindingDigest)
if nargin<3
    bindingDigest=sha256(uint8('FL-COP++ default binding'));
end
rng(seed+777,'twister'); %#ok<RNGR> % records the experiment seed; Java uses its own CSPRNG
kpg=java.security.KeyPairGenerator.getInstance('EC');
spec=java.security.spec.ECGenParameterSpec('secp256r1');
kpg.initialize(spec);
serverKP=kpg.generateKeyPair();
serverPubBytes=numel(javaBytesToUint8(serverKP.getPublic().getEncoded()));

clientKeys=cell(numClients,1);
ledger=repmat(struct('ECDHSeconds',0,'ClientPublicKeyBytes',0, ...
    'ServerPublicKeyBytes',serverPubBytes,'SigningPublicKeyBytes',0, ...
    'SignatureBytes',0,'BindingVerified',false),numClients,1);

for c=1:numClients
    t=tic;
    clientKP=kpg.generateKeyPair();

    kaClient=javax.crypto.KeyAgreement.getInstance('ECDH');
    kaClient.init(clientKP.getPrivate());
    kaClient.doPhase(serverKP.getPublic(),true);
    secretClient=javaBytesToUint8(kaClient.generateSecret());

    kaServer=javax.crypto.KeyAgreement.getInstance('ECDH');
    kaServer.init(serverKP.getPrivate());
    kaServer.doPhase(clientKP.getPublic(),true);
    secretServer=javaBytesToUint8(kaServer.generateSecret());
    if ~isequal(secretClient,secretServer)
        error('ECDH shared-secret mismatch for client %d.',c);
    end

    context=uint8(char("FL-COP++|client|"+string(c)));
    sessionKey=hkdfSHA256(secretClient,uint8('FLCOPPP-HKDF-SALT'),context,32);
    signingKP=kpg.generateKeyPair();

    bundle=struct();
    bundle.SessionKey=sessionKey;
    bundle.SigningPrivateKey=signingKP.getPrivate();
    bundle.SigningPublicKey=signingKP.getPublic();
    bundle.ClientEphemeralPublicKey=clientKP.getPublic();
    bundle.ServerEphemeralPublicKey=serverKP.getPublic();
    bundle.InitialBindingDigest=uint8(bindingDigest(:));
    clientKeys{c}=bundle;

    ledger(c).ECDHSeconds=toc(t);
    ledger(c).ClientPublicKeyBytes=numel(javaBytesToUint8(clientKP.getPublic().getEncoded()));
    ledger(c).SigningPublicKeyBytes=numel(javaBytesToUint8(signingKP.getPublic().getEncoded()));
end
end

function [signatureBytes,ok]=signAndVerifyRegistration(bundle,registrationDigest)
message=sha256([uint8('FL-COP++ enrollment-v5')'; ...
    uint8(bundle.InitialBindingDigest(:));uint8(registrationDigest(:)); ...
    javaBytesToUint8(bundle.ClientEphemeralPublicKey.getEncoded())'; ...
    javaBytesToUint8(bundle.ServerEphemeralPublicKey.getEncoded())']);
signer=java.security.Signature.getInstance('SHA256withECDSA');
signer.initSign(bundle.SigningPrivateKey);
signer.update(toJavaByteArray(message));
sig=signer.sign();
signatureBytes=javaBytesToUint8(sig);
signer.initVerify(bundle.SigningPublicKey);
signer.update(toJavaByteArray(message));
ok=signer.verify(sig);
end

function out = cryptoSelfTest(seed)
rng(seed+888,'twister');
out = struct('Passed',false,'ECDH',false,'ECDSA',false,'AESGCM',false);

kpg = java.security.KeyPairGenerator.getInstance('EC');
kpg.initialize(java.security.spec.ECGenParameterSpec('secp256r1'));
a = kpg.generateKeyPair();
b = kpg.generateKeyPair();

ka = javax.crypto.KeyAgreement.getInstance('ECDH');
ka.init(a.getPrivate()); ka.doPhase(b.getPublic(),true);
sa = javaBytesToUint8(ka.generateSecret());
kb = javax.crypto.KeyAgreement.getInstance('ECDH');
kb.init(b.getPrivate()); kb.doPhase(a.getPublic(),true);
sb = javaBytesToUint8(kb.generateSecret());
out.ECDH = isequal(sa,sb);

sig = java.security.Signature.getInstance('SHA256withECDSA');
message = uint8('FL-COP++ ephemeral-key binding self-test');
sig.initSign(a.getPrivate());
sig.update(toJavaByteArray(message));
signature = sig.sign();
sig.initVerify(a.getPublic());
sig.update(toJavaByteArray(message));
out.ECDSA = sig.verify(signature);

key = hkdfSHA256(sa,uint8('salt'),uint8('aead'),32);
nonce = secureRandomBytes(12);
cipher = javax.crypto.Cipher.getInstance('AES/GCM/NoPadding');
gcm = javax.crypto.spec.GCMParameterSpec(128,toJavaByteArray(nonce));
keySpec = javax.crypto.spec.SecretKeySpec(toJavaByteArray(key),'AES');
cipher.init(javax.crypto.Cipher.ENCRYPT_MODE,keySpec,gcm);
ct = cipher.doFinal(toJavaByteArray(message));
cipher2 = javax.crypto.Cipher.getInstance('AES/GCM/NoPadding');
gcm2 = javax.crypto.spec.GCMParameterSpec(128,toJavaByteArray(nonce));
cipher2.init(javax.crypto.Cipher.DECRYPT_MODE,keySpec,gcm2);
pt = javaBytesToUint8(cipher2.doFinal(ct));
out.AESGCM = isequal(pt,message);
out.Passed = out.ECDH && out.ECDSA && out.AESGCM;
if ~out.Passed
    error('Cryptographic self-test failed.');
end
end

function okm = hkdfSHA256(ikm,salt,info,L)
prk = hmacSHA256(salt,ikm);
T = uint8([]);
okm = uint8([]);
counter = uint8(1);
while numel(okm)<L
    T = hmacSHA256(prk,[T(:); info(:); counter(:)]);
    okm = [okm(:); T(:)]; %#ok<AGROW>
    counter = counter+1;
end
okm = okm(1:L)';
end

function digest = sha256(bytes)
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(toJavaByteArray(bytes));
digest = javaBytesToUint8(md.digest());
digest = uint8(digest(:));
end

function tag = hmacSHA256(key,data)
mac = javax.crypto.Mac.getInstance('HmacSHA256');
ks = javax.crypto.spec.SecretKeySpec(toJavaByteArray(key),'HmacSHA256');
mac.init(ks);
tag = javaBytesToUint8(mac.doFinal(toJavaByteArray(data)));
tag = uint8(tag(:));
end

function bytes = secureRandomBytes(n)
sr = java.security.SecureRandom();
bytes = javaBytesToUint8(sr.generateSeed(n));
end

function u = javaBytesToUint8(jbytes)
j = int8(jbytes);
u = typecast(j(:),'uint8');
u = u(:)';
end

function j = toJavaByteArray(bytes)
% Preserve the exact uint8 bit pattern when passing data to Java byte[].
u = uint8(bytes(:));
j = reshape(typecast(u,'int8'),1,[]);
end

%% ========================================================================
function tf = constantTimeEqual(a,b)
if numel(a)~=numel(b)
    tf = false;
    return;
end
x = uint8(a(:));
y = uint8(b(:));
acc = uint8(0);
for i = 1:numel(x)
    acc = bitor(acc,bitxor(x(i),y(i)));
end
tf = acc==0;
end

function blocks = splitBlocks(bytes,blockSize)
bytes = uint8(bytes(:));
n = max(1,ceil(numel(bytes)/blockSize));
blocks = cell(n,1);
for i = 1:n
    s = (i-1)*blockSize+1;
    e = min(i*blockSize,numel(bytes));
    if s>numel(bytes)
        blocks{i} = uint8([]);
    else
        blocks{i} = bytes(s:e);
    end
end
end

function block = getBlock(bytes,index,blockSize)
bytes = uint8(bytes(:));
s = (index-1)*blockSize+1;
e = min(index*blockSize,numel(bytes));
if s>numel(bytes)
    block = uint8([]);
else
    block = bytes(s:e);
end
end

function changed = countChangedBlocks(referenceBytes,candidateBytes,blockSize)
%COUNTCHANGEDBLOCKS Counts block positions whose byte content differs.
% The comparison is length-aware: insertion/deletion creates additional
% changed blocks rather than being silently truncated to the shorter input.
referenceBytes = uint8(referenceBytes(:));
candidateBytes = uint8(candidateBytes(:));
if blockSize<=0 || blockSize~=floor(blockSize)
    error('blockSize must be a positive integer.');
end
nReference = max(1,ceil(numel(referenceBytes)/blockSize));
nCandidate = max(1,ceil(numel(candidateBytes)/blockSize));
nBlocks = max(nReference,nCandidate);
changed = 0;
for b = 1:nBlocks
    referenceBlock = getBlock(referenceBytes,b,blockSize);
    candidateBlock = getBlock(candidateBytes,b,blockSize);
    if ~isequal(referenceBlock,candidateBlock)
        changed = changed + 1;
    end
end
end

function bytes = serializeClientData(X,Y)
% Canonical sample-wise serialization. Each record binds its stable local
% position, image bytes, and label so label changes and record reordering
% alter the committed block stream. Integers use network byte order.
n=size(X,4); imageBytes=size(X,1)*size(X,2)*size(X,3); recordBytes=4+imageBytes+1;
header=[uint8('FLC5')';u32be(size(X,1));u32be(size(X,2)); ...
    u32be(size(X,3));u32be(n)];
bytes=zeros(numel(header)+n*recordBytes,1,'uint8');
bytes(1:numel(header))=header; cursor=numel(header)+1;
for i=1:n
    img=reshape(uint8(X(:,:,:,i)),[],1);
    rec=[u32be(i);img;uint8(Y(i))];
    bytes(cursor:cursor+recordBytes-1)=rec;
    cursor=cursor+recordBytes;
end
end

function b=u32be(x)
x=uint32(x); b=zeros(4,1,'uint8');
for i=1:4, b(i)=uint8(bitand(bitshift(x,-8*(4-i)),uint32(255))); end
end

%% ========================================================================
function [microTable,baselineTable,theoryTable] = runIntegrityMicrobenchmarks(bytes,key,cfg)
% Reviewer-facing integrity experiments using the same separated client and
% server protocol used by the FL loop.
[clientState,serverRecord] = flcoppp_protocol("enroll",bytes,key, ...
    cfg.integrity.blockSize,cfg.integrity.codeDigest,cfg.integrity.configDigest,1);

attackNames = [
    "legitimate"
    "randomBlockModification"
    "targetedTailModification"
    "insertionShift"
    "deletionShift"
    "blockReordering"
    "configDigestModification"
    "partialChallengeDisclosure"
    "dualCopyEvasion"
];

rows=struct([]); theoryRows=struct([]);
for a=1:numel(attackNames)
    name=attackNames(a);
    detected=false(cfg.integrity.microbenchTrials,1);
    clientMs=zeros(cfg.integrity.microbenchTrials,1);
    serverMs=zeros(cfg.integrity.microbenchTrials,1);
    totalMs=zeros(cfg.integrity.microbenchTrials,1);
    wire=zeros(cfg.integrity.microbenchTrials,1);
    trainingChanged=zeros(cfg.integrity.microbenchTrials,1);
    visibleChanged=zeros(cfg.integrity.microbenchTrials,1);
    exactPerTrial=nan(cfg.integrity.microbenchTrials,1);
    approxPerTrial=nan(cfg.integrity.microbenchTrials,1);

    localServer=serverRecord;
    for t=1:cfg.integrity.microbenchTrials
        [challenge,localServer,cb]=flcoppp_protocol("challenge",localServer,t, ...
            cfg.integrity.challengeCount);
        [~,proofSource,trainM,visibleM,exactP,approxP] = ...
            makeTamperedBytesV5(bytes,name,cfg.integrity.blockSize,challenge,cfg,t);
        trainingChanged(t)=trainM;
        visibleChanged(t)=visibleM;
        exactPerTrial(t)=exactP;
        approxPerTrial(t)=approxP;

        tt=tic;
        tc=tic;
        if name=="configDigestModification"
            badConfig=cfg.integrity.configDigest; badConfig(1)=bitxor(badConfig(1),uint8(1));
            [packet,~]=flcoppp_protocol("prove",clientState,proofSource,challenge, ...
                cfg.integrity.codeDigest,badConfig);
        else
            [packet,~]=flcoppp_protocol("prove",clientState,proofSource,challenge);
        end
        clientMs(t)=1000*toc(tc);
        ts=tic;
        [pass,localServer,~]=flcoppp_protocol("verify",localServer,challenge,packet);
        serverMs(t)=1000*toc(ts);
        totalMs(t)=1000*toc(tt);
        wire(t)=cb+packet.WireBytes;
        detected(t)=~pass;
    end

    empirical=mean(detected);
    r=struct();
    r.Attack=name;
    r.Trials=cfg.integrity.microbenchTrials;
    r.Detected=sum(detected);
    r.DetectionRate=empirical;
    [r.DetectionCI_Low,r.DetectionCI_High]=wilsonCI(sum(detected),cfg.integrity.microbenchTrials,cfg.statistics.confidenceLevel);
    r.AcceptanceRate=1-empirical;
    if name=="legitimate"
        r.FalsePositiveRate=empirical;
        r.FalseNegativeRate=NaN;
    else
        r.FalsePositiveRate=NaN;
        r.FalseNegativeRate=1-empirical;
    end
    r.MedianTrainingCopyChangedBlocks=median(trainingChanged);
    r.MedianVerifierVisibleChangedBlocks=median(visibleChanged);
    r.TotalBlocks=double(serverRecord.NumBlocks);
    r.MedianClientProofMs=median(clientMs);
    r.MedianServerVerifyMs=median(serverMs);
    r.MedianEndToEndMs=median(totalMs);
    r.MeanEndToEndMs=mean(totalMs);
    r.StdEndToEndMs=std(totalMs);
    r.MeanTotalWireBytes=mean(wire);
    r.ExactDetectionProbability=mean(exactPerTrial,'omitnan');
    r.ApproxDetectionProbability=mean(approxPerTrial,'omitnan');
    r.EmpiricalMinusExact=empirical-r.ExactDetectionProbability;
    rows=appendStructRow(rows,r);

    tr=struct();
    tr.Attack=name;
    tr.DataFraction=1.0;
    tr.DataBytes=numel(bytes);
    tr.TotalBlocks=double(serverRecord.NumBlocks);
    tr.ModifiedBlocksVisibleToVerifier=median(visibleChanged);
    tr.Challenges=min(cfg.integrity.challengeCount,double(serverRecord.NumBlocks));
    tr.EmpiricalDetection=empirical;
    tr.ExactDetection=mean(exactPerTrial,'omitnan');
    tr.IndependentApproximation=mean(approxPerTrial,'omitnan');
    tr.AbsoluteApproximationError=abs(tr.ExactDetection-tr.IndependentApproximation);
    tr.RelativeApproximationError=safeDivide(tr.AbsoluteApproximationError,tr.ExactDetection);
    theoryRows=appendStructRow(theoryRows,tr);
end
microTable=struct2table(rows);
theoryTable=struct2table(theoryRows);
baselineTable=benchmarkIntegrityBaselinesV5(bytes,key,cfg);

% Add the full q-by-tampering parameter sweep to the theory table.
B=baselineTable(baselineTable.Scheme=="FLCOPPP",:);
extra=struct([]);
for i=1:height(B)
    approx=1-(1-B.ModifiedBlocks(i)/B.TotalBlocks(i))^B.Challenges(i);
    e=struct(); e.Attack="parameterSweep";
    e.DataFraction=B.DataFraction(i); e.DataBytes=B.DataBytes(i);
    e.TotalBlocks=B.TotalBlocks(i);
    e.ModifiedBlocksVisibleToVerifier=B.ModifiedBlocks(i);
    e.Challenges=B.Challenges(i);
    e.EmpiricalDetection=B.DetectionRate(i);
    e.ExactDetection=B.ExactDetectionProbability(i);
    e.IndependentApproximation=approx;
    e.AbsoluteApproximationError=abs(e.ExactDetection-approx);
    e.RelativeApproximationError=safeDivide(e.AbsoluteApproximationError,e.ExactDetection);
    extra=appendStructRow(extra,e);
end
if ~isempty(extra), theoryTable=[theoryTable;struct2table(extra)]; end
end

function [trainingCopy,proofSource,trainingChanged,visibleChanged,exactP,approxP] = ...
    makeTamperedBytesV5(bytes,name,blockSize,challenge,cfg,trial)
trainingCopy=uint8(bytes(:)); proofSource=trainingCopy;
N=max(1,ceil(numel(bytes)/blockSize));
m=max(1,ceil(0.05*N));
stream=RandStream('mt19937ar','Seed',900000+trial+double(sum(char(name))));
q=numel(challenge.Indices);

switch name
    case "legitimate"
        exactP=0; approxP=0;

    case "randomBlockModification"
        blocks=randperm(stream,N,min(m,N));
        trainingCopy=modifyFirstByteOfBlocks(trainingCopy,blocks,blockSize,uint8(1));
        proofSource=trainingCopy;
        exactP=exactDetectionProbability(N,numel(blocks),q);
        approxP=1-(1-numel(blocks)/N)^q;

    case "targetedTailModification"
        blocks=max(1,N-m+1):N;
        trainingCopy=modifyFirstByteOfBlocks(trainingCopy,blocks,blockSize,uint8(3));
        proofSource=trainingCopy;
        exactP=exactDetectionProbability(N,numel(blocks),q);
        approxP=1-(1-numel(blocks)/N)^q;

    case "insertionShift"
        pos=max(2,round(numel(trainingCopy)/3));
        inserted=uint8(floor(256*rand(stream,128,1)));
        trainingCopy=[trainingCopy(1:pos);inserted;trainingCopy(pos+1:end)];
        proofSource=trainingCopy;
        exactP=1; approxP=NaN; % deterministic byte-length binding

    case "deletionShift"
        pos=max(2,round(numel(trainingCopy)/3));
        e=min(numel(trainingCopy),pos+127);
        trainingCopy(pos:e)=[];
        proofSource=trainingCopy;
        exactP=1; approxP=NaN;

    case "blockReordering"
        if N>=3
            b1=2; b2=N;
            blk1=getBlock(trainingCopy,b1,blockSize);
            blk2=getBlock(trainingCopy,b2,blockSize);
            s1=(b1-1)*blockSize+1; s2=(b2-1)*blockSize+1;
            len=min(numel(blk1),numel(blk2));
            tmp=trainingCopy(s1:s1+len-1);
            trainingCopy(s1:s1+len-1)=trainingCopy(s2:s2+len-1);
            trainingCopy(s2:s2+len-1)=tmp;
        end
        proofSource=trainingCopy;
        mm=countChangedBlocks(bytes,proofSource,blockSize);
        exactP=exactDetectionProbability(N,mm,q);
        approxP=1-(1-mm/N)^q;

    case "configDigestModification"
        proofSource=trainingCopy;
        exactP=1; approxP=NaN;

    case "partialChallengeDisclosure"
        actual=double(challenge.Indices(:)');
        knownCount=min(q,max(0,round(cfg.integrity.partialKnowledgeFraction*q)));
        if knownCount>0
            known=actual(randperm(stream,q,knownCount));
        else
            known=[];
        end
        candidates=setdiff(1:N,known);
        blocks=candidates(randperm(stream,numel(candidates),min(m,numel(candidates))));
        trainingCopy=modifyFirstByteOfBlocks(trainingCopy,blocks,blockSize,uint8(7));
        proofSource=trainingCopy;
        unknownQ=q-knownCount;
        universe=N-knownCount;
        exactP=conditionalDisclosureDetection(universe,numel(blocks),unknownQ);
        approxP=1-(1-numel(blocks)/max(universe,1))^unknownQ;

    case "dualCopyEvasion"
        pos=max(2,round(numel(trainingCopy)/2));
        trainingCopy(pos)=bitxor(trainingCopy(pos),uint8(15));
        proofSource=uint8(bytes(:));
        exactP=0; approxP=0; % verifier sees an unchanged retained copy

    otherwise
        error("Unknown integrity attack: %s",name);
end
trainingChanged=countChangedBlocks(bytes,trainingCopy,blockSize);
visibleChanged=countChangedBlocks(bytes,proofSource,blockSize);
end

function out=modifyFirstByteOfBlocks(in,blocks,blockSize,mask)
out=uint8(in(:));
for b=blocks
    pos=min(numel(out),(b-1)*blockSize+1);
    out(pos)=bitxor(out(pos),mask);
end
end

function p=conditionalDisclosureDetection(universe,modified,unknownChallenges)
if unknownChallenges<=0 || modified<=0, p=0; return; end
p=exactDetectionProbability(universe,modified,unknownChallenges);
end

function T=benchmarkIntegrityBaselinesV5(bytes,key,cfg)
% Accuracy-overhead matrix for representative integrity baselines.
rows=struct([]);
for df=cfg.integrity.baselineDataFractions
    L=max(1024,floor(df*numel(bytes)));
    base=uint8(bytes(1:min(L,numel(bytes))));
    N=max(1,ceil(numel(base)/cfg.integrity.blockSize));
    % Full-state AUTHENTICATED HMAC baseline (not plain SHA-256).
    baseDigest=hmacSHA256(key,base);
    baseBlockHashes=cell(N,1);
    for b=1:N, baseBlockHashes{b}=sha256(getBlock(base,b,cfg.integrity.blockSize)); end

    for frac=cfg.integrity.baselineTamperFractions
        m=min(N,max(1,ceil(frac*N)));
        for qRequested=cfg.integrity.baselineChallengeCounts
            q=min(qRequested,N);
            schemes=["FullStateHMAC";"SampledBlockHashes";"AllBlockMerkleProofs";"FLCOPPP"];
            for si=1:numel(schemes)
                scheme=schemes(si);
                if scheme=="FullStateHMAC" && qRequested~=cfg.integrity.baselineChallengeCounts(1), continue; end
                if scheme=="AllBlockMerkleProofs" && qRequested~=cfg.integrity.baselineChallengeCounts(1), continue; end

                detects=false(cfg.integrity.microbenchTrials,1);
                clientMs=zeros(cfg.integrity.microbenchTrials,1);
                serverMs=zeros(cfg.integrity.microbenchTrials,1);
                comm=zeros(cfg.integrity.microbenchTrials,1);
                readBytes=zeros(cfg.integrity.microbenchTrials,1);
                [clientState,serverRecord]=flcoppp_protocol("enroll",base,key, ...
                    cfg.integrity.blockSize,cfg.integrity.codeDigest,cfg.integrity.configDigest,2);

                for t=1:cfg.integrity.microbenchTrials
                    stream=RandStream('mt19937ar','Seed',700000+t+round(10000*frac)+qRequested);
                    blocks=randperm(stream,N,m);
                    tampered=modifyFirstByteOfBlocks(base,blocks,cfg.integrity.blockSize,uint8(1));

                    switch scheme
                        case "FullStateHMAC"
                            nonce=secureRandomBytes(16);
                            tc=tic; currentDigest=hmacSHA256(key,tampered); clientMs(t)=1000*toc(tc);
                            ts=tic; detects(t)=~constantTimeEqual(currentDigest,baseDigest); serverMs(t)=1000*toc(ts);
                            comm(t)=numel(nonce)+32+5;
                            readBytes(t)=numel(tampered);

                        case "SampledBlockHashes"
                            [challenge,serverRecord,cb]=flcoppp_protocol("challenge",serverRecord,t,q);
                            idx=double(challenge.Indices);
                            tc=tic;
                            currentHashes=cell(q,1); responseBody=uint8([]);
                            rb=0;
                            for j=1:q
                                blk=getBlock(tampered,idx(j),cfg.integrity.blockSize);
                                currentHashes{j}=sha256(blk); rb=rb+numel(blk);
                                responseBody=[responseBody;u32be(idx(j));currentHashes{j}(:)]; %#ok<AGROW>
                            end
                            responseToken=hmacSHA256(key,[challenge.Digest(:);responseBody(:)]);
                            clientMs(t)=1000*toc(tc);
                            ts=tic; mismatch=false;
                            expectedToken=hmacSHA256(key,[challenge.Digest(:);responseBody(:)]);
                            tokenOK=constantTimeEqual(responseToken,expectedToken);
                            for j=1:q
                                mismatch=mismatch || ~constantTimeEqual(currentHashes{j},baseBlockHashes{idx(j)});
                            end
                            serverMs(t)=1000*toc(ts); detects(t)=mismatch || ~tokenOK;
                            comm(t)=cb+numel(responseBody)+numel(responseToken)+5; readBytes(t)=rb;

                        case "AllBlockMerkleProofs"
                            [challenge,serverRecord,cb]=flcoppp_protocol("challenge",serverRecord,t,N);
                            tc=tic; [packet,~]=flcoppp_protocol("prove",clientState,tampered,challenge); clientMs(t)=1000*toc(tc);
                            ts=tic; [pass,serverRecord,~]=flcoppp_protocol("verify",serverRecord,challenge,packet); serverMs(t)=1000*toc(ts);
                            detects(t)=~pass; comm(t)=cb+packet.WireBytes; readBytes(t)=numel(tampered);

                        case "FLCOPPP"
                            [challenge,serverRecord,cb]=flcoppp_protocol("challenge",serverRecord,t,q);
                            tc=tic; [packet,~]=flcoppp_protocol("prove",clientState,tampered,challenge); clientMs(t)=1000*toc(tc);
                            ts=tic; [pass,serverRecord,~]=flcoppp_protocol("verify",serverRecord,challenge,packet); serverMs(t)=1000*toc(ts);
                            detects(t)=~pass; comm(t)=cb+packet.WireBytes;
                            rb=0; for j=1:q, rb=rb+numel(getBlock(tampered,double(challenge.Indices(j)),cfg.integrity.blockSize)); end
                            readBytes(t)=rb;
                    end
                end

                r=struct();
                r.Scheme=scheme;
                r.DataFraction=df;
                r.DataBytes=numel(base);
                r.TotalBlocks=N;
                r.ModifiedFraction=frac;
                r.ModifiedBlocks=m;
                if scheme=="FullStateHMAC" || scheme=="AllBlockMerkleProofs"
                    r.Challenges=N;
                else
                    r.Challenges=q;
                end
                r.Trials=cfg.integrity.microbenchTrials;
                r.DetectionRate=mean(detects);
                [r.DetectionCI_Low,r.DetectionCI_High]=wilsonCI(sum(detects),cfg.integrity.microbenchTrials,cfg.statistics.confidenceLevel);
                r.ExactDetectionProbability=schemeExactDetection(scheme,N,m,q);
                r.MedianClientMs=median(clientMs);
                r.MedianServerMs=median(serverMs);
                r.MeanCommunicationBytes=mean(comm);
                r.MeanClientReadBytes=mean(readBytes);
                r.ContentExposure=false;
                if scheme=="SampledBlockHashes"
                    r.ServerStorageBytes=32*N;
                    r.ClientComputationOrder="O(qB)";
                elseif scheme=="FullStateHMAC"
                    r.ServerStorageBytes=32;
                    r.ClientComputationOrder="O(L)";
                elseif scheme=="AllBlockMerkleProofs"
                    r.ServerStorageBytes=32;
                    r.ClientComputationOrder="O(N log N)";
                else
                    r.ServerStorageBytes=32;
                    r.ClientComputationOrder="O(q log N)";
                end
                rows=appendStructRow(rows,r);
            end
        end
    end
end
T=struct2table(rows);
end

function p=schemeExactDetection(scheme,N,m,q)
if scheme=="FullStateHMAC" || scheme=="AllBlockMerkleProofs"
    p=double(m>0);
else
    p=exactDetectionProbability(N,m,q);
end
end

function p = exactDetectionProbability(N,m,q)
if m<=0, p=0; return; end
q=min(q,N);
if q>N-m, p=1; return; end
logMiss=logChoose(N-m,q)-logChoose(N,q);
p=min(max(1-exp(logMiss),0),1);
end

function y = logChoose(n,k)
if k<0 || k>n, y=-Inf;
else, y=gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1);
end
end

%% ========================================================================
function T = summarizeFinalMetrics(R,clientT,cfg)
methods=unique(R.Method,'stable'); attacks=unique(R.Attack,'stable'); rows=struct([]);
for a=1:numel(attacks)
    for m=1:numel(methods)
        mask=R.Attack==attacks(a) & R.Method==methods(m);
        seeds=unique(R.Seed(mask));
        ca=nan(numel(seeds),1); asr=nan(numel(seeds),1);
        bytes=nan(numel(seeds),1); seconds=nan(numel(seeds),1);
        convergence=nan(numel(seeds),1); targets=nan(numel(seeds),1);
        seedTPR=nan(numel(seeds),1); seedFPR=nan(numel(seeds),1);

        for si=1:numel(seeds)
            seed=seeds(si);
            x=sortrows(R(mask & R.Seed==seed,:),'Round');
            ca(si)=lastNonNaN(x.CleanAccuracy);
            asr(si)=lastNonNaN(x.AttackSuccessRate);
            bytes(si)=sum(x.TotalBytes); seconds(si)=sum(x.RoundSeconds)+sum(x.EnrollmentSeconds);

            ref=sortrows(R(R.Seed==seed & R.Attack=="clean" & R.Method=="FedAvg",:),'Round');
            refFinal=lastNonNaN(ref.CleanAccuracy);
            if ~isnan(refFinal)
                targets(si)=cfg.statistics.convergenceFractionOfCleanFedAvg*refFinal;
                convergence(si)=convergenceRoundToTarget(x.Round,x.CleanAccuracy,targets(si));
            end

            cs=clientT(clientT.Seed==seed & clientT.Attack==attacks(a) & ...
                clientT.Method==methods(m),:);
            active=cs.AttackActive; rejected=cs.Rejected; benign=~active;
            [seedTPR(si),seedFPR(si),~,~]=detectionMetrics( ...
                sum(rejected & active),sum(~rejected & active), ...
                sum(rejected & benign),sum(~rejected & benign));
        end

        cmask=clientT.Attack==attacks(a) & clientT.Method==methods(m);
        TP=sum(clientT.Rejected(cmask) & clientT.AttackActive(cmask));
        FN=sum(~clientT.Rejected(cmask) & clientT.AttackActive(cmask));
        benign=~clientT.AttackActive(cmask);
        FP=sum(clientT.Rejected(cmask) & benign);
        TN=sum(~clientT.Rejected(cmask) & benign);
        [tpr,fpr,precision,f1]=detectionMetrics(TP,FN,FP,TN);
        [tprLo,tprHi]=wilsonCI(TP,TP+FN,cfg.statistics.confidenceLevel);
        [fprLo,fprHi]=wilsonCI(FP,FP+TN,cfg.statistics.confidenceLevel);

        r=struct(); r.Dataset=string(cfg.datasetName); r.Attack=attacks(a); r.Method=methods(m); r.Seeds=numel(seeds);
        [r.CA_Mean,r.CA_SD,r.CA_CI_Low,r.CA_CI_High]=meanCI(ca,cfg.statistics.confidenceLevel);
        [r.ASR_Mean,r.ASR_SD,r.ASR_CI_Low,r.ASR_CI_High]=meanCI(asr,cfg.statistics.confidenceLevel);
        r.TPR=tpr; r.TPR_CI_Low=tprLo; r.TPR_CI_High=tprHi;
        r.FPR=fpr; r.FPR_CI_Low=fprLo; r.FPR_CI_High=fprHi;
        [r.TPR_SeedMean,r.TPR_SeedSD,r.TPR_SeedCI_Low,r.TPR_SeedCI_High]=meanCI(seedTPR,cfg.statistics.confidenceLevel);
        [r.FPR_SeedMean,r.FPR_SeedSD,r.FPR_SeedCI_Low,r.FPR_SeedCI_High]=meanCI(seedFPR,cfg.statistics.confidenceLevel);
        r.Precision=precision; r.F1=f1; r.TP=TP; r.FN=FN; r.FP=FP; r.TN=TN;
        r.ConvergenceTargetCA_Mean=mean(targets,'omitnan');
        r.ConvergenceRoundMean=mean(convergence,'omitnan');
        r.ConvergenceRoundSD=std(convergence,'omitnan');
        r.TotalCommunicationMB_Mean=mean(bytes,'omitnan')/1024^2;
        r.TotalCommunicationMB_SD=std(bytes,'omitnan')/1024^2;
        r.TotalWallClockSeconds_Mean=mean(seconds,'omitnan');
        r.TotalWallClockSeconds_SD=std(seconds,'omitnan');
        rows=appendStructRow(rows,r);
    end
end
T=struct2table(rows);
end

function T = summarizeDetection(C,cfg)
methods = unique(C.Method,'stable');
attacks = unique(C.Attack,'stable');
rows = struct([]);
for a = 1:numel(attacks)
    for m = 1:numel(methods)
        mask = C.Attack==attacks(a) & C.Method==methods(m);
        active = C.AttackActive(mask);
        rejected = C.Rejected(mask);
        malicious = C.IsMalicious(mask);
        TP = sum(rejected & active);
        FN = sum(~rejected & active);
        benign = ~active;
        FP = sum(rejected & benign);
        TN = sum(~rejected & benign);
        [tpr,fpr,p,f1] = detectionMetrics(TP,FN,FP,TN);
        [tl,th] = wilsonCI(TP,TP+FN,cfg.statistics.confidenceLevel);
        [fl,fh] = wilsonCI(FP,FP+TN,cfg.statistics.confidenceLevel);
        r = struct('Attack',attacks(a),'Method',methods(m), ...
            'TP',TP,'FN',FN,'FP',FP,'TN',TN,'TPR',tpr,'TPR_CI_Low',tl, ...
            'TPR_CI_High',th,'FPR',fpr,'FPR_CI_Low',fl,'FPR_CI_High',fh, ...
            'Precision',p,'F1',f1);
        rows = appendStructRow(rows,r);
    end
end
T = struct2table(rows);
end

function T = summarizeDetectorAttribution(C,cfg)
methods=unique(C.Method,'stable');
attacks=unique(C.Attack,'stable');
rows=struct([]);
for a=1:numel(attacks)
    for m=1:numel(methods)
        x=C(C.Attack==attacks(a) & C.Method==methods(m),:);
        active=x.AttackActive;
        benign=~active;

        preRejected=~x.PreGatePass;
        preTP=sum(preRejected & active);
        preFN=sum(~preRejected & active);
        preFP=sum(preRejected & benign);
        preTN=sum(~preRejected & benign);
        [preTPR,preFPR,~,~]=detectionMetrics(preTP,preFN,preFP,preTN);

        eligible=x.Trained;
        behFlag=x.BehaviorFlag & eligible;
        behActive=active & eligible;
        behBenign=benign & eligible;
        behTP=sum(behFlag & behActive);
        behFN=sum(~behFlag & behActive);
        behFP=sum(behFlag & behBenign);
        behTN=sum(~behFlag & behBenign);
        [behTPR,behFPR,~,~]=detectionMetrics(behTP,behFN,behFP,behTN);

        combined=x.Rejected;
        TP=sum(combined & active); FN=sum(~combined & active);
        FP=sum(combined & benign); TN=sum(~combined & benign);
        [combTPR,combFPR,~,~]=detectionMetrics(TP,FN,FP,TN);

        r=struct();
        r.Attack=attacks(a); r.Method=methods(m);
        r.PreGateTPR=preTPR; r.PreGateFPR=preFPR;
        r.PreGateTP=preTP; r.PreGateFN=preFN; r.PreGateFP=preFP; r.PreGateTN=preTN;
        r.BehaviorTPR=behTPR; r.BehaviorFPR=behFPR;
        r.BehaviorTP=behTP; r.BehaviorFN=behFN; r.BehaviorFP=behFP; r.BehaviorTN=behTN;
        r.CombinedTPR=combTPR; r.CombinedFPR=combFPR;
        r.CombinedTP=TP; r.CombinedFN=FN; r.CombinedFP=FP; r.CombinedTN=TN;
        [r.PreGateFPR_CI_Low,r.PreGateFPR_CI_High]=wilsonCI(preFP,preFP+preTN,cfg.statistics.confidenceLevel);
        [r.BehaviorFPR_CI_Low,r.BehaviorFPR_CI_High]=wilsonCI(behFP,behFP+behTN,cfg.statistics.confidenceLevel);
        rows = appendStructRow(rows,r);
    end
end
T=struct2table(rows);
end

function T=summarizeCryptoFailureReasons(C)
reasons=unique(C.PreGateFailureReason,'stable'); rows=struct([]);
for i=1:numel(reasons)
    mask=C.PreGateFailureReason==reasons(i);
    r=struct(); r.FailureReason=reasons(i); r.ClientRounds=sum(mask);
    r.BenignClientRounds=sum(mask & ~C.AttackActive);
    r.ActiveMaliciousClientRounds=sum(mask & C.AttackActive);
    r.CryptographicRejects=sum(mask & ~C.PreGatePass);
    rows=appendStructRow(rows,r);
end
T=struct2table(rows);
end

function T = summarizeCommunication(R)
methods=unique(R.Method,'stable'); attacks=unique(R.Attack,'stable'); rows=struct([]);
for a=1:numel(attacks)
    for m=1:numel(methods)
        x=R(R.Attack==attacks(a) & R.Method==methods(m),:);
        seeds=unique(x.Seed); n=numel(seeds);
        D=zeros(n,1); U=zeros(n,1); C=zeros(n,1); P=zeros(n,1); I=zeros(n,1); E=zeros(n,1); Tt=zeros(n,1); CR=zeros(n,1);
        for k=1:n
            z=x(x.Seed==seeds(k),:);
            D(k)=sum(z.DownlinkBytes); U(k)=sum(z.UpdateBytes);
            if ismember('ChallengeBytes',z.Properties.VariableNames), C(k)=sum(z.ChallengeBytes); end
            if ismember('ProofBytes',z.Properties.VariableNames), P(k)=sum(z.ProofBytes); end
            I(k)=sum(z.IntegrityBytes); E(k)=sum(z.EnrollmentBytes); Tt(k)=sum(z.TotalBytes);
            CR(k)=mean(z.MeanCompressionRatio,'omitnan');
        end
        r=struct(); r.Attack=attacks(a); r.Method=methods(m); r.Seeds=n;
        r.DownlinkMB=mean(D)/1024^2; r.DownlinkMB_SD=std(D)/1024^2;
        r.UpdateMB=mean(U)/1024^2; r.UpdateMB_SD=std(U)/1024^2;
        r.ChallengeMB=mean(C)/1024^2; r.ChallengeMB_SD=std(C)/1024^2;
        r.ProofMB=mean(P)/1024^2; r.ProofMB_SD=std(P)/1024^2;
        r.IntegrityMB=mean(I)/1024^2; r.IntegrityMB_SD=std(I)/1024^2;
        r.EnrollmentMB=mean(E)/1024^2; r.EnrollmentMB_SD=std(E)/1024^2;
        r.TotalMB=mean(Tt)/1024^2; r.TotalMB_SD=std(Tt)/1024^2;
        r.IntegrityOverheadPercent=mean(100*(I+E)./max(Tt,1));
        r.MeanCompressionRatio=mean(CR,'omitnan');
        rows=appendStructRow(rows,r);
    end
end
T=struct2table(rows);
end

function T = summarizeRuntime(R)
methods=unique(R.Method,'stable'); attacks=unique(R.Attack,'stable'); rows=struct([]);
for a=1:numel(attacks)
    for m=1:numel(methods)
        x=R(R.Attack==attacks(a) & R.Method==methods(m),:);
        seeds=unique(x.Seed); n=numel(seeds);
        total=zeros(n,1); meanRound=zeros(n,1); integ=zeros(n,1); local=zeros(n,1); beh=zeros(n,1); agg=zeros(n,1);
        for k=1:n
            z=x(x.Seed==seeds(k),:);
            total(k)=sum(z.RoundSeconds)+sum(z.EnrollmentSeconds); meanRound(k)=mean(z.RoundSeconds);
            integ(k)=sum(z.IntegritySeconds); local(k)=sum(z.LocalTrainingSeconds);
            beh(k)=sum(z.BehaviorSeconds); agg(k)=sum(z.AggregationSeconds);
        end
        r=struct(); r.Attack=attacks(a); r.Method=methods(m); r.Seeds=n;
        r.TotalSeconds=mean(total); r.TotalSeconds_SD=std(total);
        r.MeanRoundSeconds=mean(meanRound); r.MeanRoundSeconds_SD=std(meanRound);
        r.MedianRoundSeconds=median(x.RoundSeconds);
        r.EnrollmentSeconds=mean(arrayfun(@(seed) sum(x.EnrollmentSeconds(x.Seed==seed)),seeds));
        r.IntegritySeconds=mean(integ); r.LocalTrainingSeconds=mean(local);
        r.BehaviorSeconds=mean(beh); r.AggregationSeconds=mean(agg);
        rows=appendStructRow(rows,r);
    end
end
T=struct2table(rows);
end

function T = buildComplexityTable(cfg,modelVector,clientIndices)
nBlocks = zeros(numel(clientIndices),1);
for i = 1:numel(clientIndices)
    estimatedBytes = 20 + numel(clientIndices{i})*(4+32*32*3+1);
    nBlocks(i) = ceil(estimatedBytes/cfg.integrity.blockSize);
end
rows = {
    "Enrollment Merkle-HMAC","O(N)","O(N) client, O(1) server","N = data blocks"
    "PRF challenge generation","O(q)","O(q)","nonce-bound sampling without replacement"
    "Per-round proof generation","O(q log N)","O(N) retained tree","q = challenged blocks"
    "Per-round proof verification","O(q log N)","O(1) server root/state","AEAD, token, length, indices, paths"
    "FedAvg aggregation","O(PM)","O(PM)","P parameters, M updates"
    "Coordinate median","O(P M log M)","O(PM)","coordinate sorting"
    "Krum","O(P M^2)","O(M^2)","pairwise update distances"
    "RFA","O(I P M)","O(PM)","I Weiszfeld iterations"
    "Top-k compression","O(P log P)","O(P)","reference implementation uses sort"
    "Uniform quantization","O(K)","O(K)","K retained coefficients"
    };
T = cell2table(rows,'VariableNames',{'Component','TimeComplexity','SpaceComplexity','Definition'});
T.MeanClientBlocks = repmat(mean(nBlocks),height(T),1);
T.ModelParameters = repmat(numel(modelVector),height(T),1);
end

function [meanVal,sdVal,lo,hi] = meanCI(x,confidence)
x = x(~isnan(x));
if isempty(x)
    meanVal=NaN; sdVal=NaN; lo=NaN; hi=NaN; return;
end
meanVal=mean(x); sdVal=std(x);
if numel(x)==1
    lo=NaN; hi=NaN; return;
end
alpha=1-confidence;
tcrit = tCriticalApprox(numel(x)-1,1-alpha/2);
half=tcrit*sdVal/sqrt(numel(x));
lo=meanVal-half; hi=meanVal+half;
end

function t = tCriticalApprox(df,p)
% 0.975 critical values; p is retained for interface clarity.
if p < 0.97 || p > 0.98
    z = 1.96;
else
    z = 1.96;
end
table975 = [12.706 4.303 3.182 2.776 2.571 2.447 2.365 2.306 2.262 ...
    2.228 2.201 2.179 2.160 2.145 2.131 2.120 2.110 2.101 2.093 2.086 ...
    2.080 2.074 2.069 2.064 2.060 2.056 2.052 2.048 2.045 2.042];
if df>=1 && df<=numel(table975)
    t=table975(df);
else
    t=z;
end
end

function [lo,hi] = wilsonCI(k,n,confidence)
if n<=0
    lo=NaN; hi=NaN; return;
end
z=1.95996398454005;
if abs(confidence-0.95)>1e-9
    z=1.95996398454005;
end
phat=k/n;
den=1+z^2/n;
center=(phat+z^2/(2*n))/den;
half=z/den*sqrt(phat*(1-phat)/n+z^2/(4*n^2));
lo=max(0,center-half); hi=min(1,center+half);
end

function r = convergenceRoundToTarget(rounds,ca,target)
valid=~isnan(ca);
rounds=rounds(valid); ca=ca(valid);
if isempty(ca) || isnan(target), r=NaN; return; end
idx=find(ca>=target,1,'first');
if isempty(idx), r=NaN; else, r=rounds(idx); end
end

%% ========================================================================
function A = runAblationSuite(cfg,XTrain,YTrain,XTest,YTest,XRef,YRef,outputDir)
% One-factor-at-a-time ablations. A single seed is used by default to keep
% the full reviewer suite tractable; the selected final settings are then
% evaluated with all seeds in the main experiment.
seed=cfg.statistics.seeds(1);
base=cfg;
base.fl.numRounds=cfg.ablation.numRounds;
base.statistics.seeds=seed;
base.suite.methods="FLCOPPP";
base.suite.attacks="modelReplacementBackdoor";

rows=struct([]);
factors = {
    "numClients", cfg.ablation.numClients
    "challengeCount", cfg.ablation.challengeCounts
    "blockSize", cfg.ablation.blockSizes
    "nonIIDAlpha", cfg.ablation.nonIIDAlpha
    "maliciousFraction", cfg.ablation.maliciousFractions
    "sparsity", cfg.ablation.sparsities
    "quantizationBits", cfg.ablation.quantizationBits
    };

for f=1:size(factors,1)
    factor=string(factors{f,1});
    values=factors{f,2};
    for v=1:numel(values)
        c=base;
        value=values(v);
        switch factor
            case "numClients"
                c.fl.numClients=value;
                c.fl.clientsPerRound=max(4,round(0.20*value));
                c.fl.minSamplesPerClient=min(c.fl.minSamplesPerClient, ...
                    floor((numel(YTrain)-c.data.validationCount)/(2*value)));
            case "challengeCount", c.integrity.challengeCount=value;
            case "blockSize", c.integrity.blockSize=value;
            case "nonIIDAlpha", c.fl.nonIIDAlpha=value;
            case "maliciousFraction", c.attack.maliciousFraction=value;
            case "sparsity", c.compression.sparsity=value;
            case "quantizationBits", c.compression.bits=value;
        end

        rng(seed,'twister');
        [refIdx,poolIdx]=stratifiedHoldout(YTrain,c.data.validationCount,seed);
        Xr=XTrain(:,:,:,refIdx); Yr=YTrain(refIdx);
        parts=dirichletPartition(YTrain(poolIdx),c.fl.numClients,c.fl.nonIIDAlpha, ...
            c.fl.minSamplesPerClient,seed);
        clientIndices=cell(c.fl.numClients,1);
        for i=1:c.fl.numClients, clientIndices{i}=poolIdx(parts{i}); end
        schedule=makeClientSchedule(c.fl.numRounds,c.fl.numClients,c.fl.clientsPerRound,seed+10000);
        malicious=chooseMaliciousClients(c.fl.numClients,c.attack.maliciousFraction,seed+20000);
        net=buildCIFARNetwork(c); [vec,meta]=networkToVector(net);
        c = attachProtocolDigests(c);
        [keys,keyLedger]=makeClientSessionKeys(c.fl.numClients,seed,c.integrity.bindingDigest);
        ckpt=fullfile(outputDir,'checkpoints',sprintf('ablation_%s_%g.mat',factor,value));
        [R,~,C,~]=runSingleExperiment(c,seed,"modelReplacementBackdoor","FLCOPPP", ...
            XTrain,YTrain,XTest,YTest,Xr,Yr,clientIndices,schedule,malicious,net, ...
            vec,meta,keys,keyLedger,ckpt, ...
            makeCheckpointSignature(c,seed,"modelReplacementBackdoor","FLCOPPP"));

        final=R(end,:);
        TP=sum(C.Rejected & C.AttackActive);
        FN=sum(~C.Rejected & C.AttackActive);
        benign=~C.AttackActive;
        FP=sum(C.Rejected & benign);
        TN=sum(~C.Rejected & benign);
        [tpr,fpr,~,~]=detectionMetrics(TP,FN,FP,TN);

        r=struct();
        r.Factor=factor; r.Setting=string(value); r.Value=value;
        r.CleanAccuracy=final.CleanAccuracy;
        r.AttackSuccessRate=final.AttackSuccessRate;
        r.TPR=tpr; r.FPR=fpr;
        r.TotalCommunicationMB=sum(R.TotalBytes)/1024^2;
        r.TotalSeconds=sum(R.RoundSeconds);
        rows = appendStructRow(rows,r);
    end
end

% Component ablation isolates the pre-gate, behavioral probe, trust
% weighting, and compression rather than attributing all gain to one bundle.
componentMethods = cfg.ablation.componentMethods;
c=base;
c.suite.attacks="modelReplacementBackdoor";
rng(seed,'twister');
[refIdx,poolIdx]=stratifiedHoldout(YTrain,c.data.validationCount,seed);
Xr=XTrain(:,:,:,refIdx); Yr=YTrain(refIdx);
parts=dirichletPartition(YTrain(poolIdx),c.fl.numClients,c.fl.nonIIDAlpha, ...
    c.fl.minSamplesPerClient,seed);
clientIndices=cell(c.fl.numClients,1);
for i=1:c.fl.numClients, clientIndices{i}=poolIdx(parts{i}); end
schedule=makeClientSchedule(c.fl.numRounds,c.fl.numClients,c.fl.clientsPerRound,seed+10000);
malicious=chooseMaliciousClients(c.fl.numClients,c.attack.maliciousFraction,seed+20000);
rng(seed+30000,'twister');
baseNet=buildCIFARNetwork(c); [baseVec,baseMeta]=networkToVector(baseNet);
c = attachProtocolDigests(c);
[keys,keyLedger]=makeClientSessionKeys(c.fl.numClients,seed,c.integrity.bindingDigest);

for i=1:numel(componentMethods)
    method=componentMethods(i);
    ckpt=fullfile(outputDir,'checkpoints',"ablation_component_"+sanitizeName(method)+".mat");
    [R,~,C,~]=runSingleExperiment(c,seed,"modelReplacementBackdoor",method, ...
        XTrain,YTrain,XTest,YTest,Xr,Yr,clientIndices,schedule,malicious,baseNet, ...
        baseVec,baseMeta,keys,keyLedger,ckpt, ...
        makeCheckpointSignature(c,seed,"modelReplacementBackdoor",method));
    final=R(end,:);
    TP=sum(C.Rejected & C.AttackActive);
    FN=sum(~C.Rejected & C.AttackActive);
    benign=~C.AttackActive;
    FP=sum(C.Rejected & benign);
    TN=sum(~C.Rejected & benign);
    [tpr,fpr,~,~]=detectionMetrics(TP,FN,FP,TN);
    r=struct();
    r.Factor="component";
    r.Setting=method;
    r.Value=NaN;
    r.CleanAccuracy=final.CleanAccuracy;
    r.AttackSuccessRate=final.AttackSuccessRate;
    r.TPR=tpr; r.FPR=fpr;
    r.TotalCommunicationMB=sum(R.TotalBytes)/1024^2;
    r.TotalSeconds=sum(R.RoundSeconds);
    rows = appendStructRow(rows,r);
end

A=struct2table(rows);
end

%% ========================================================================
function exportAllTables(outputDir,cfg,varargin)
names = {
    'RoundMetrics','TrustMetrics','ClientDecisions','RunLedger','FinalSummary', ...
    'DetectionSummary','DetectorAttribution','CryptoFailureAttribution','CommunicationSummary','RuntimeSummary','ComplexityTable', ...
    'IntegrityMicrobenchmark','IntegrityBaselines','TheoryEmpirical','AblationResults'
    };
tablesDir=fullfile(outputDir,'tables');
excelFile=fullfile(tablesDir,'FLCOPPP_complete_results.xlsx');
if isfile(excelFile), delete(excelFile); end

for i=1:numel(varargin)
    T=varargin{i};
    if ~istable(T), continue; end
    name=names{i};
    if cfg.outputs.saveCSV
        writetable(T,fullfile(tablesDir,name+".csv"));
    end
    if cfg.outputs.saveExcel
        try
            writetable(T,excelFile,'Sheet',name);
        catch ME
            warning('Excel export failed for %s: %s',name,ME.message);
        end
    end
end
end

function generateFigures(outputDir,cfg,R,Trust,Final,Detection,Comm,Runtime,Micro,Baselines,Theory,Ablation)
figDir=fullfile(outputDir,'figures');
set(groot,'defaultAxesFontName','Times New Roman');
set(groot,'defaultTextFontName','Times New Roman');

attacks=unique(R.Attack,'stable');
methods=unique(R.Method,'stable');

for a=1:numel(attacks)
    attack=attacks(a);
    f=figure('Visible','off','Color','w');
    hold on;
    for m=1:numel(methods)
        x=R(R.Attack==attack & R.Method==methods(m),:);
        if isempty(x), continue; end
        rounds=unique(x.Round);
        y=zeros(numel(rounds),1);
        for k=1:numel(rounds)
            y(k)=mean(x.CleanAccuracy(x.Round==rounds(k)),'omitnan');
        end
        plot(rounds,100*y,'LineWidth',1.5,'DisplayName',methods(m));
    end
    xlabel('Federated round'); ylabel('Clean accuracy (%)');
    title("Clean accuracy under " + attack,'Interpreter','none');
    grid on; legend('Location','bestoutside','Interpreter','none');
    exportFigurePair(f,figDir,"CA_"+sanitizeName(attack),cfg); close(f);

    if any(~isnan(R.AttackSuccessRate(R.Attack==attack)))
        f=figure('Visible','off','Color','w'); hold on;
        for m=1:numel(methods)
            x=R(R.Attack==attack & R.Method==methods(m),:);
            if isempty(x), continue; end
            rounds=unique(x.Round); y=zeros(numel(rounds),1);
            for k=1:numel(rounds)
                y(k)=mean(x.AttackSuccessRate(x.Round==rounds(k)),'omitnan');
            end
            plot(rounds,100*y,'LineWidth',1.5,'DisplayName',methods(m));
        end
        xlabel('Federated round'); ylabel('Attack success rate (%)');
        title("ASR under " + attack,'Interpreter','none');
        grid on; legend('Location','bestoutside','Interpreter','none');
        exportFigurePair(f,figDir,"ASR_"+sanitizeName(attack),cfg); close(f);
    end
end

f=figure('Visible','off','Color','w');
bar(categorical(Final.Method+" | "+Final.Attack),100*Final.CA_Mean);
ylabel('Final clean accuracy (%)'); xtickangle(45); grid on;
title('Final clean accuracy by method and attack');
exportFigurePair(f,figDir,'Final_CA',cfg); close(f);

valid=~isnan(Final.ASR_Mean);
if any(valid)
    f=figure('Visible','off','Color','w');
    bar(categorical(Final.Method(valid)+" | "+Final.Attack(valid)),100*Final.ASR_Mean(valid));
    ylabel('Final ASR (%)'); xtickangle(45); grid on;
    title('Final attack success rate');
    exportFigurePair(f,figDir,'Final_ASR',cfg); close(f);
end

f=figure('Visible','off','Color','w');
Y=[100*Detection.TPR,100*Detection.FPR];
bar(categorical(Detection.Method+" | "+Detection.Attack),Y);
ylabel('Rate (%)'); legend({'TPR','FPR'},'Location','best');
xtickangle(45); grid on; title('Client-round detection performance');
exportFigurePair(f,figDir,'TPR_FPR',cfg); close(f);

f=figure('Visible','off','Color','w');
Y=[Comm.DownlinkMB,Comm.UpdateMB,Comm.IntegrityMB+Comm.EnrollmentMB];
bar(categorical(Comm.Method+" | "+Comm.Attack),Y,'stacked');
ylabel('Communication (MB)'); legend({'Downlink','Updates','Integrity metadata'});
xtickangle(45); grid on; title('Measured communication breakdown');
exportFigurePair(f,figDir,'Communication_breakdown',cfg); close(f);

f=figure('Visible','off','Color','w');
Y=[Runtime.EnrollmentSeconds,Runtime.LocalTrainingSeconds,Runtime.IntegritySeconds,Runtime.BehaviorSeconds,Runtime.AggregationSeconds];
bar(categorical(Runtime.Method+" | "+Runtime.Attack),Y,'stacked');
ylabel('Wall-clock time (s)'); legend({'Enrollment','Local training','Integrity','Behavior','Aggregation'});
xtickangle(45); grid on; title('Runtime breakdown');
exportFigurePair(f,figDir,'Runtime_breakdown',cfg); close(f);

f=figure('Visible','off','Color','w');
bar(categorical(Micro.Attack),100*Micro.DetectionRate);
ylabel('Detection rate (%)'); xtickangle(35); ylim([0 105]); grid on;
title('Merkle-HMAC integrity microbenchmarks');
exportFigurePair(f,figDir,'Integrity_attacks',cfg); close(f);

f=figure('Visible','off','Color','w'); hold on;
mask=Baselines.DataFraction==max(Baselines.DataFraction) & ...
    abs(Baselines.ModifiedFraction-0.05)==min(abs(Baselines.ModifiedFraction-0.05));
B=Baselines(mask,:);
scatter(B.MeanCommunicationBytes,100*B.DetectionRate,45,'filled');
for i=1:height(B)
    text(B.MeanCommunicationBytes(i),100*B.DetectionRate(i), ...
        "  "+B.Scheme(i)+" q="+string(B.Challenges(i)),'Interpreter','none');
end
xlabel('Measured communication bytes'); ylabel('Detection rate (%)');
grid on; title('Integrity accuracy-overhead comparison');
exportFigurePair(f,figDir,'Integrity_baselines',cfg); close(f);

f=figure('Visible','off','Color','w'); hold on;
Tplot=Theory(Theory.Attack=="parameterSweep" & ...
    Theory.DataFraction==max(Theory.DataFraction),:);
scatter(Tplot.ExactDetection,Tplot.EmpiricalDetection,45,Tplot.Challenges,'filled');
plot([0 1],[0 1],'k--','LineWidth',1.2,'DisplayName','Identity');
xlabel('Exact detection probability'); ylabel('Empirical detection rate');
xlim([0 1]); ylim([0 1]); grid on; colorbar;
title('Empirical validation of exact detection probability');
exportFigurePair(f,figDir,'Theory_vs_empirical',cfg); close(f);

if ~isempty(Trust)
    f=figure('Visible','off','Color','w'); hold on;
    combos=unique(Trust.Method+" | "+Trust.Attack,'stable');
    for i=1:numel(combos)
        mask=(Trust.Method+" | "+Trust.Attack)==combos(i);
        x=Trust(mask,:);
        rounds=unique(x.Round);
        gap=zeros(numel(rounds),1);
        for k=1:numel(rounds), gap(k)=mean(x.TrustGap(x.Round==rounds(k)),'omitnan'); end
        plot(rounds,gap,'LineWidth',1.3,'DisplayName',combos(i));
    end
    xlabel('Round'); ylabel('Benign minus malicious trust');
    grid on; legend('Location','bestoutside','Interpreter','none');
    title('Trust separation trajectories');
    exportFigurePair(f,figDir,'Trust_gap',cfg); close(f);
end

if ~isempty(Ablation)
    factors=unique(Ablation.Factor,'stable');
    for i=1:numel(factors)
        x=Ablation(Ablation.Factor==factors(i),:);
        if factors(i)=="component"
            f=figure('Visible','off','Color','w');
            Y=[100*x.CleanAccuracy,100*x.AttackSuccessRate];
            bar(categorical(x.Setting),Y);
            ylabel('Rate (%)'); legend({'CA','ASR'},'Location','best');
            xtickangle(30); grid on; title('FL-COP++ component ablation');
            exportFigurePair(f,figDir,'Ablation_components',cfg); close(f);
        else
            f=figure('Visible','off','Color','w');
            yyaxis left
            plot(x.Value,100*x.CleanAccuracy,'o-','LineWidth',1.5); ylabel('CA (%)');
            yyaxis right
            plot(x.Value,100*x.AttackSuccessRate,'s--','LineWidth',1.5); ylabel('ASR (%)');
            xlabel(factors(i),'Interpreter','none'); grid on;
            title("Ablation: "+factors(i),'Interpreter','none');
            exportFigurePair(f,figDir,"Ablation_"+sanitizeName(factors(i)),cfg); close(f);
        end
    end
end
end

function exportFigurePair(f,dirPath,name,cfg)
exportgraphics(f,fullfile(dirPath,name+".png"),'Resolution',cfg.outputs.figureResolution);
exportgraphics(f,fullfile(dirPath,name+".pdf"),'ContentType','vector');
end

function writeTextSummary(outputDir,cfg,Final,Detection,Comm,Runtime,Micro)
fid=fopen(fullfile(outputDir,'experiment_summary.txt'),'w','n','UTF-8');
cleanup=onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'FL-COP++ CIFAR-10 experiment summary\n');
fprintf(fid,'Generated: %s\n\n',char(datetime('now')));
fprintf(fid,'Profile: %s\nClients: %d\nRounds: %d\nSeeds: %s\n\n', ...
    cfg.profile,cfg.fl.numClients,cfg.fl.numRounds,mat2str(cfg.statistics.seeds));

for i=1:height(Final)
    fprintf(fid,'%s | %s\n',Final.Method(i),Final.Attack(i));
    fprintf(fid,'  CA: %.4f ± %.4f, 95%% CI [%.4f, %.4f]\n', ...
        Final.CA_Mean(i),Final.CA_SD(i),Final.CA_CI_Low(i),Final.CA_CI_High(i));
    fprintf(fid,'  ASR: %.4f ± %.4f, 95%% CI [%.4f, %.4f]\n', ...
        Final.ASR_Mean(i),Final.ASR_SD(i),Final.ASR_CI_Low(i),Final.ASR_CI_High(i));
    fprintf(fid,'  TPR: %.4f | FPR: %.4f | communication: %.2f MB\n\n', ...
        Final.TPR(i),Final.FPR(i),Final.TotalCommunicationMB_Mean(i));
end

fprintf(fid,'Important interpretation:\n');
fprintf(fid,['The Merkle-HMAC pre-gate verifies consistency with a committed byte state. ' ...
    'It does not prove that the committed data were used for training and cannot identify ' ...
    'poisoning already present before enrollment. The dualCopyEvasion microbenchmark ' ...
    'makes this limitation explicit.\n']);
end

%% ========================================================================
function writeSystemLedger(cfg,cifarDir,XTrain,YTrain,XTest,YTest,classNames,outputDir)
ledger=struct();
ledger.GeneratedAt=char(datetime('now','TimeZone','local'));
ledger.MATLABVersion=version;
ledger.Release=version('-release');
ledger.Computer=computer;
ledger.OS=string(char(java.lang.System.getProperty('os.name'))) + " " + string(char(java.lang.System.getProperty('os.version')));
ledger.Architecture=char(java.lang.System.getProperty('os.arch'));
ledger.JavaVersion=char(java.lang.System.getProperty('java.version'));
try, ledger.LogicalCores=feature('numcores'); catch, ledger.LogicalCores=NaN; end
try
    [userMem,systemMem]=memory;
    ledger.MATLABMemoryAvailableBytes=userMem.MemAvailableAllArrays;
    ledger.SystemMemoryBytes=systemMem.PhysicalMemory.Total;
catch
    ledger.MATLABMemoryAvailableBytes=NaN; ledger.SystemMemoryBytes=NaN;
end
ledger.GPURequested=cfg.execution.useGPU;
ledger.GPUAvailable=shouldUseGPU(cfg);
if ledger.GPUAvailable
    try
        g=gpuDevice;
        ledger.GPUName=g.Name; ledger.GPUComputeCapability=g.ComputeCapability;
        ledger.GPUTotalMemoryBytes=g.TotalMemory;
    catch
        ledger.GPUName="available but not queried";
        ledger.GPUComputeCapability=""; ledger.GPUTotalMemoryBytes=NaN;
    end
else
    ledger.GPUName="none"; ledger.GPUComputeCapability=""; ledger.GPUTotalMemoryBytes=0;
end
ledger.DatasetRoot=char(cifarDir);
ledger.Dataset=string(cfg.datasetName);
ledger.ReferenceSetAssumption=string(cfg.data.referenceSetAssumption);
ledger.TrainingImages=size(XTrain,4); ledger.TestImages=size(XTest,4);
ledger.ServerReferenceImages=cfg.data.validationCount;
ledger.NormalizationSource=cfg.data.normalizationSource;
ledger.ChannelMean=cfg.data.channelMean; ledger.ChannelStd=cfg.data.channelStd;
ledger.Classes=numel(unique(YTrain)); ledger.ClassNames=cellstr(classNames);
ledger.TrainingLabelCounts=histcounts(double(YTrain),-0.5:1:(cfg.data.numClasses-0.5));
ledger.TestLabelCounts=histcounts(double(YTest),-0.5:1:(cfg.data.numClasses-0.5));
txt=jsonencode(ledger,'PrettyPrint',true);
fid=fopen(fullfile(outputDir,'system_and_dataset_ledger.json'),'w','n','UTF-8');
fwrite(fid,txt,'char'); fclose(fid);
end

function writeProtocolTestReport(report,outputDir)
txt=jsonencode(report,'PrettyPrint',true);
fid=fopen(fullfile(outputDir,'protocol_self_test.json'),'w','n','UTF-8');
fwrite(fid,txt,'char'); fclose(fid);
end

function cfg = attachProtocolDigests(cfg)
if isfield(cfg,'execution') && isfield(cfg.execution,'codeFile')
    primary=char(cfg.execution.codeFile);
else
    primary=[mfilename('fullpath') '.m'];
end
codeDir=fileparts(primary);
codeFiles={primary,fullfile(codeDir,'flcoppp_protocol.m'), ...
    fullfile(codeDir,'flcoppp_config.m'), ...
    fullfile(codeDir,'flcoppp_apply_femnist_config.m'), ...
    fullfile(codeDir,'flcoppp_load_femnist.m'), ...
    fullfile(codeDir,'flcoppp_writer_reference_split.m'), ...
    fullfile(codeDir,'flcoppp_export_reviewer_audit.m')};
bundle=uint8([]);
for i=1:numel(codeFiles)
    if ~isfile(codeFiles{i}), error('Bound code file not found: %s',codeFiles{i}); end
    fid=fopen(codeFiles{i},'r');
    if fid<0, error('Cannot read bound code file: %s',codeFiles{i}); end
    bytes=fread(fid,Inf,'*uint8'); fclose(fid);
    [~,nm,ext]=fileparts(codeFiles{i});
    nameBytes=unicode2native(string(nm)+string(ext),'UTF-8');
    bundle=[bundle;uint8(nameBytes(:));uint8(0);sha256(bytes)]; %#ok<AGROW>
end
codeDigest=sha256(bundle);
boundNames=strings(numel(codeFiles),1);
for i=1:numel(codeFiles)
    [~,nm,ext]=fileparts(codeFiles{i}); boundNames(i)=string(nm)+string(ext);
end
cfg.execution.boundCodeFiles=boundNames;

cfgForDigest=cfg;
if isfield(cfgForDigest,'integrity')
    removable={'codeDigest','configDigest','bindingDigest', ...
        'codeDigestHex','configDigestHex','bindingDigestHex'};
    for i=1:numel(removable)
        if isfield(cfgForDigest.integrity,removable{i})
            cfgForDigest.integrity=rmfield(cfgForDigest.integrity,removable{i});
        end
    end
end
configBytes=unicode2native(jsonencode(cfgForDigest),'UTF-8');
configDigest=sha256(uint8(configBytes));
bindingDigest=sha256([codeDigest(:);configDigest(:)]);

cfg.integrity.codeDigest=codeDigest;
cfg.integrity.configDigest=configDigest;
cfg.integrity.bindingDigest=bindingDigest;
cfg.integrity.codeDigestHex=bytesToHex(codeDigest);
cfg.integrity.configDigestHex=bytesToHex(configDigest);
cfg.integrity.bindingDigestHex=bytesToHex(bindingDigest);
end

function h = bytesToHex(bytes)
h=lower(reshape(dec2hex(uint8(bytes),2).',1,[]));
end

function saveConfiguration(cfg,outputDir)
save(fullfile(outputDir,'experiment_config.mat'),'cfg');
txt=jsonencode(cfg,'PrettyPrint',true);
fid=fopen(fullfile(outputDir,'experiment_config.json'),'w','n','UTF-8');
fwrite(fid,txt,'char'); fclose(fid);
end

function validateEnvironment(cfg)
if cfg.profile~="protocol_test" && exist('dlnetwork','file')~=2
    error('Deep Learning Toolbox is required: dlnetwork was not found.');
end
if ~usejava('jvm')
    error('MATLAB Java/JVM support is required for SHA-256, HMAC, ECDH, ECDSA, and AES-GCM.');
end
end

function tf = shouldUseGPU(cfg)
tf=false;
if ~cfg.execution.useGPU, return; end
try
    if exist('canUseGPU','file')==2
        tf=canUseGPU;
    elseif exist('gpuDeviceCount','file')==2
        tf=gpuDeviceCount>0;
    end
catch
    tf=false;
end
end

function folder = resolveCifarFolder(rootPath)
rootPath=string(rootPath);
candidates=[
    rootPath
    fullfile(rootPath,"cifar-10-batches-mat")
    fullfile(rootPath,"cifar-10-matlab","cifar-10-batches-mat")
];
for i=1:numel(candidates)
    if isfile(fullfile(candidates(i),"data_batch_1.mat")) && ...
            isfile(fullfile(candidates(i),"test_batch.mat"))
        folder=candidates(i);
        return;
    end
end
error(['CIFAR-10 MAT files were not found. Checked:\n%s\n' ...
    'Expected data_batch_1.mat ... data_batch_5.mat and test_batch.mat.'], ...
    strjoin(candidates,newline));
end

function [XTrain,YTrain,XTest,YTest,classNames] = loadCIFAR10(folder)
XTrain=zeros(32,32,3,50000,'uint8');
YTrain=zeros(50000,1,'uint8');
offset=0;
for b=1:5
    S=load(fullfile(folder,sprintf('data_batch_%d.mat',b)),'data','labels');
    n=size(S.data,1);
    X=reshape(S.data',[32,32,3,n]);
    X=permute(X,[2 1 3 4]);
    XTrain(:,:,:,offset+1:offset+n)=X;
    YTrain(offset+1:offset+n)=uint8(S.labels);
    offset=offset+n;
end
S=load(fullfile(folder,'test_batch.mat'),'data','labels');
n=size(S.data,1);
X=reshape(S.data',[32,32,3,n]);
XTest=permute(X,[2 1 3 4]);
YTest=uint8(S.labels);

metaFile=fullfile(folder,'batches.meta.mat');
if isfile(metaFile)
    M=load(metaFile);
    if isfield(M,'label_names')
        classNames=string(M.label_names);
    else
        classNames="class_"+string(0:9);
    end
else
    classNames="class_"+string(0:9);
end
end

function mu = computeChannelMean(X)
mu=zeros(1,3);
for c=1:3
    z=single(X(:,:,c,:))/255;
    mu(c)=mean(z(:));
end
end

function sigma = computeChannelStd(X,mu)
sigma=zeros(1,3);
for c=1:3
    z=single(X(:,:,c,:))/255;
    sigma(c)=sqrt(mean((z(:)-mu(c)).^2));
    sigma(c)=max(sigma(c),1e-6);
end
end

function [refIdx,poolIdx] = stratifiedHoldout(Y,count,seed)
stream=RandStream('mt19937ar','Seed',seed+123);
classes=unique(Y)';
perClass=floor(count/numel(classes));
refIdx=[];
for c=classes
    idx=find(Y==c);
    k=min(perClass,numel(idx));
    refIdx=[refIdx;idx(randperm(stream,numel(idx),k))]; %#ok<AGROW>
end
remaining=count-numel(refIdx);
if remaining>0
    candidates=setdiff((1:numel(Y))',refIdx);
    refIdx=[refIdx;candidates(randperm(stream,numel(candidates),remaining))];
end
refIdx=refIdx(:);
poolIdx=setdiff((1:numel(Y))',refIdx);
end

function parts = dirichletPartition(labels,numClients,alpha,minSamples,seed)
stream=RandStream('mt19937ar','Seed',seed+456);
parts=cell(numClients,1);
classes=unique(labels)';

for c=classes
    idx=find(labels==c);
    idx=idx(randperm(stream,numel(idx)));
    g=gammaSamples(stream,alpha,numClients);
    p=g/sum(g);
    raw=p*numel(idx);
    counts=floor(raw);
    rem=numel(idx)-sum(counts);
    [~,ord]=sort(raw-counts,'descend');
    counts(ord(1:rem))=counts(ord(1:rem))+1;
    pos=1;
    for k=1:numClients
        if counts(k)>0
            parts{k}=[parts{k};idx(pos:pos+counts(k)-1)]; %#ok<AGROW>
            pos=pos+counts(k);
        end
    end
end

% Deterministic minimum-size rebalancing.
sizes=cellfun(@numel,parts);
while any(sizes<minSamples)
    [~,poor]=min(sizes);
    [~,donor]=max(sizes);
    if sizes(donor)<=minSamples
        break;
    end
    take=min(minSamples-sizes(poor),sizes(donor)-minSamples);
    donorOrder=randperm(stream,sizes(donor),take);
    moved=parts{donor}(donorOrder);
    parts{donor}(donorOrder)=[];
    parts{poor}=[parts{poor};moved];
    sizes=cellfun(@numel,parts);
end

for k=1:numClients
    parts{k}=parts{k}(randperm(stream,numel(parts{k})));
end
end

function g = gammaSamples(stream,shape,n)
% Marsaglia-Tsang gamma sampler using the supplied RandStream.
g=zeros(1,n);
for i=1:n
    a=shape;
    if a<1
        u=max(rand(stream),realmin);
        g(i)=gammaOne(stream,a+1)*u^(1/a);
    else
        g(i)=gammaOne(stream,a);
    end
end
end

function x = gammaOne(stream,a)
d=a-1/3;
c=1/sqrt(9*d);
while true
    z=randn(stream);
    v=(1+c*z)^3;
    if v<=0, continue; end
    u=rand(stream);
    if u < 1-0.0331*z^4 || log(u) < 0.5*z^2+d*(1-v+log(v))
        x=d*v;
        return;
    end
end
end

function schedule = makeClientSchedule(rounds,numClients,clientsPerRound,seed)
stream=RandStream('mt19937ar','Seed',seed);
schedule=zeros(rounds,clientsPerRound);
for r=1:rounds
    schedule(r,:)=randperm(stream,numClients,clientsPerRound);
end
end

function mask = chooseMaliciousClients(numClients,fraction,seed)
stream=RandStream('mt19937ar','Seed',seed);
n=max(1,round(numClients*fraction));
mask=false(numClients,1);
mask(randperm(stream,numClients,n))=true;
end

function ensureDir(p)
if ~isfolder(p), mkdir(p); end
end

function y = safeDivide(a,b)
if b==0, y=NaN; else, y=a/b; end
end

function x = lastNonNaN(v)
idx=find(~isnan(v),1,'last');
if isempty(idx), x=NaN; else, x=v(idx); end
end

function s = sanitizeName(x)
s=regexprep(char(string(x)),'[^A-Za-z0-9_]','_');
end

function s = fmtRate(x)
if isnan(x)
    s='N/A';
else
    s=sprintf('%.3f',x);
end
end

function s = fmtPercentOrNA(x)
if isnan(x)
    s='   N/A ';
else
    s=sprintf('%6.2f%%',100*x);
end
end


function signature = makeCheckpointSignature(cfg,seed,attackName,methodName)
signature = strjoin([
    "schema=" + string(cfg.execution.checkpointSchemaVersion)
    "code=" + string(cfg.integrity.codeDigestHex)
    "config=" + string(cfg.integrity.configDigestHex)
    "seed=" + string(seed)
    "attack=" + string(attackName)
    "method=" + string(methodName)
],"|");
end

function rows = appendStructRow(rows,row)
%APPENDSTRUCTROW Safely appends a scalar structure even when a stale
%checkpoint was created with a different field order or an older schema.
if isempty(rows)
    rows = row;
    return;
end

if ~isscalar(row)
    error('appendStructRow expects a scalar row structure.');
end

oldFields = fieldnames(rows);
newFields = fieldnames(row);
allFields = unique([oldFields; newFields],'stable');

% Add newly introduced fields to every existing row.
for i = 1:numel(allFields)
    f = allFields{i};
    if ~isfield(rows,f)
        defaultValue = defaultForStructField(row.(f));
        [rows.(f)] = deal(defaultValue);
    end
end

% Add fields retained only by an older checkpoint to the new row.
for i = 1:numel(allFields)
    f = allFields{i};
    if ~isfield(row,f)
        row.(f) = defaultForStructField(rows(1).(f));
    end
end

rows = orderfields(rows,allFields);
row = orderfields(row,allFields);
rows(end+1) = row;
end

function value = defaultForStructField(example)
if isstring(example)
    value = string(missing);
elseif ischar(example)
    value = '';
elseif islogical(example)
    value = false(size(example));
elseif isnumeric(example)
    if isfloat(example)
        value = nan(size(example),'like',example);
    else
        value = zeros(size(example),'like',example);
    end
elseif iscell(example)
    value = cell(size(example));
elseif iscategorical(example)
    value = categorical(missing);
elseif isdatetime(example)
    value = NaT(size(example));
elseif isduration(example)
    value = seconds(nan(size(example)));
else
    value = [];
end
end
