function flcoppp_export_reviewer_audit(outputDir,R,C,ledger,cfg)
%FLCOPPP_EXPORT_REVIEWER_AUDIT Ground reviewer statistics in raw run rows.
% No midpoint-from-min/max estimates, fabricated SDs, or assumed benign N.
out=fullfile(outputDir,'tables');
methods=unique(R.Method,'stable'); attacks=unique(R.Attack,'stable');
seeds=unique(R.Seed); rows=struct([]);
for i=1:numel(methods)
    for j=1:numel(attacks)
        for k=1:numel(seeds)
            x=R(R.Method==methods(i) & R.Attack==attacks(j) & R.Seed==seeds(k),:);
            if isempty(x), continue; end
            x=sortrows(x,'Round');
            ca=x.CleanAccuracy; asr=x.AttackSuccessRate;
            ca=ca(~isnan(ca)); asr=asr(~isnan(asr));
            row=struct(); row.Dataset=string(cfg.datasetName);
            row.Method=methods(i);row.Attack=attacks(j);row.Seed=seeds(k);
            row.Rounds=numel(unique(x.Round));
            row.FinalCA=NaN; row.FinalASR=NaN;
            if ~isempty(ca),row.FinalCA=ca(end);end
            if ~isempty(asr),row.FinalASR=asr(end);end
            row.TotalDownlinkBytes=sum(x.DownlinkBytes);
            row.TotalUpdateBytes=sum(x.UpdateBytes);
            row.TotalChallengeBytes=sum(x.ChallengeBytes);
            row.TotalProofBytes=sum(x.ProofBytes);
            row.TotalIntegrityBytes=sum(x.IntegrityBytes);
            row.TotalEnrollmentBytes=sum(x.EnrollmentBytes);
            row.TotalBytes=sum(x.TotalBytes);
            row.RoundSeconds=sum(x.RoundSeconds);
            assert(row.TotalBytes==row.TotalDownlinkBytes+row.TotalUpdateBytes+ ...
                row.TotalIntegrityBytes+row.TotalEnrollmentBytes, ...
                'Failed communication audit, method %s attack %s seed %d', ...
                char(methods(i)),char(attacks(j)),seeds(k));
            assert(row.TotalIntegrityBytes==row.TotalChallengeBytes+row.TotalProofBytes);
            rows=[rows;row]; %#ok<AGROW>
        end
    end
end
if ~isempty(rows)
    writetable(struct2table(rows),fullfile(out,'Reviewer_SeedResolved_Results.csv'));
end
% Do not infer benign exposure across different attacks/methods/datasets:
% explicitly count ALL selected clean client-rounds per dataset and method.
if ~isempty(C)
    clean=C(C.Attack=="clean",:);
    methods=unique(clean.Method,'stable'); counts=struct([]);
    for j=1:numel(methods)
        z=clean(clean.Method==methods(j),:);
        r=struct();r.Dataset=string(cfg.datasetName);r.Method=methods(j);
        r.CleanClientRounds=height(z);
        r.ValidCleanPreGateOpportunities=sum(z.ChallengesUsed>0);
        r.CryptographicFalseRejections=sum(~z.PreGatePass & z.ChallengesUsed>0);
        r.BehavioralFalseFlags=sum(z.BehaviorFlag & z.Trained);
        r.ActiveMaliciousClientRounds=sum(z.AttackActive);
        r.ExpectedCleanClientRounds=cfg.fl.numRounds*cfg.fl.clientsPerRound*numel(unique(z.Seed));
        assert(r.CleanClientRounds==r.ExpectedCleanClientRounds, ...
            'Incomplete clean run matrix; cannot claim expected benign opportunities.');
        counts=[counts;r]; %#ok<AGROW>
    end
    writetable(struct2table(counts),fullfile(out,'Reviewer_Benign_Denominator_Audit.csv'));
    if any(C.ChallengesUsed>0)
        primary=C.ChallengesUsed(C.ChallengesUsed>0);
        if ~cfg.integrity.adaptiveProbeAllocation
            % A client with N<q enrolled blocks has min(N,q) challenges:
            % the fixed-q assumption only applies when N >= q.
            assert(all(primary<=cfg.integrity.challengeCount), ...
                'Fixed-budget primary q exceeded configured q');
        end
    end
end
% Full per-run ledger preserves parameter choices and exact setup.
if ~isempty(ledger)
    writetable(ledger,fullfile(out,'Reviewer_Run_Parameter_Ledger.csv'));
end
end
