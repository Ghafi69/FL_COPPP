function [refIdx,poolIdx] = flcoppp_writer_reference_split(writerIDs,requested,seed,numClients,minPerClient)
%FLCOPPP_WRITER_REFERENCE_SPLIT Disjoint trusted server reference holdout.
% Stratified by writer, with deterministic per-seed sampling. Guarantees
% each client retains minPerClient training samples.
writerIDs=double(writerIDs(:));
stream=RandStream('mt19937ar','Seed',seed+123);
capacities=zeros(numClients,1); indices=cell(numClients,1);
for c=1:numClients
    indices{c}=find(writerIDs==c);
    capacities(c)=max(0,numel(indices{c})-minPerClient);
end
if requested>sum(capacities)
    error('Cannot reserve %d clean reference samples without violating minimum writer sample counts.',requested);
end
quotas=zeros(numClients,1);
for k=1:requested
    % Prefer writers with fewer selected holdout images, maintaining balance.
    choices=find(quotas<capacities);
    [~,j]=min(quotas(choices));
    quotas(choices(j))=quotas(choices(j))+1;
end
refIdx=zeros(requested,1); cursor=1;
for c=1:numClients
    n=quotas(c);
    if n>0
        take=indices{c}(randperm(stream,numel(indices{c}),n));
        refIdx(cursor:cursor+n-1)=take; cursor=cursor+n;
    end
end
refIdx=sort(refIdx);
poolIdx=setdiff((1:numel(writerIDs))',refIdx);
assert(isempty(intersect(refIdx,poolIdx)));
end
