function [XTrain,YTrain,XTest,YTest,classNames,writerIDs,selectedWriters] = flcoppp_load_femnist(root,cfg)
%FLCOPPP_LOAD_FEMNIST Read LEAF FEMNIST train/test JSON shards by writer.
% No synthetic records, test-to-train mixing, or re-labeling. The 50 writers
% with the largest eligible training sets (tie-broken by writer ID) are
% chosen deterministically and documented in the output ledger. This
% explicit sampling protocol may differ from the original R1 experiments.
root = string(root);
trainDir = findSplit(root,"train"); testDir = findSplit(root,"test");
train = readSplit(trainDir); test = readSplit(testDir);
trainKeys = string(keys(train)); testKeys = string(keys(test));
common = intersect(trainKeys,testKeys);
if isempty(common), error('No LEAF writer IDs are shared by train and test.'); end
counts = zeros(numel(common),1);
for k=1:numel(common)
    entry=train(char(common(k)));
    counts(k)=numel(entry.y);
end
minCount = max(cfg.data.femnistMinUserSamples, cfg.fl.minSamplesPerClient+1);
eligible = find(counts>=minCount);
if numel(eligible)<cfg.fl.numClients
    error('Need %d eligible FEMNIST writers, found %d (min train images=%d).', ...
        cfg.fl.numClients,numel(eligible),minCount);
end
% sort stable identifier first, then descending size; tie-break IDs asc.
[~,nameOrder]=sort(common(eligible)); eligible=eligible(nameOrder);
[~,countOrder]=sort(counts(eligible),'descend'); eligible=eligible(countOrder);
selectedWriters=common(eligible(1:cfg.fl.numClients));
trainImages=cell(numel(selectedWriters),1); testImages=trainImages;
trainLabels=trainImages; testLabels=trainImages;
writerCells=trainImages;
for c=1:numel(selectedWriters)
    k=char(selectedWriters(c)); a=train(k); b=test(k);
    trainImages{c}=a.X; testImages{c}=b.X;
    trainLabels{c}=a.y; testLabels{c}=b.y;
    writerCells{c}=repmat(uint16(c),numel(a.y),1);
end
XTrain=cat(4,trainImages{:}); XTest=cat(4,testImages{:});
YTrain=vertcat(trainLabels{:}); YTest=vertcat(testLabels{:});
writerIDs=vertcat(writerCells{:});
classNames="class_"+string((0:61)');
fprintf('FEMNIST: %d writers, %d train samples, %d test samples.\n', ...
    numel(selectedWriters),numel(YTrain),numel(YTest));
end

function folder=findSplit(root,split)
candidates=[fullfile(root,split),fullfile(root,'data',split), ...
    fullfile(root,'femnist','data',split)];
folder="";
for i=1:numel(candidates)
    if isfolder(candidates(i)) && ~isempty(dir(fullfile(candidates(i),'*.json')))
        folder=candidates(i); return;
    end
end
error('LEAF FEMNIST %s split not found at %s; expected JSON shards.',split,root);
end

function map=readSplit(folder)
map=containers.Map('KeyType','char','ValueType','any');
shards=dir(fullfile(folder,'*.json'));
[~,order]=sort(string({shards.name})); shards=shards(order);
for f=1:numel(shards)
    contents=jsondecode(fileread(fullfile(shards(f).folder,shards(f).name)));
    if ~isfield(contents,'users') || ~isfield(contents,'user_data')
        error('Invalid LEAF JSON file %s (missing users/user_data).',shards(f).name);
    end
    users=string(contents.users);
    for j=1:numel(users)
        id=char(users(j)); field=matlab.lang.makeValidName(id);
        if ~isfield(contents.user_data,field)
            error('Missing writer %s in %s. JSON key conversion must be resolved.',id,shards(f).name);
        end
        e=contents.user_data.(field);
        y=flattenLabels(e.y);
        X=reshapeImages(e.x,numel(y));
        if any(y>61) || any(y<0), error('Invalid 62-class FEMNIST label for %s.',id); end
        if isKey(map,id)
            prev=map(id); e2=struct('X',cat(4,prev.X,X),'y',[prev.y;y]);
        else
            e2=struct('X',X,'y',y);
        end
        map(id)=e2;
    end
end
end

function y=flattenLabels(raw)
if iscell(raw)
    y=cellfun(@double,raw(:));
else
    y=double(raw(:));
end
if ~all(isfinite(y)) || any(y~=round(y)), error('Invalid FEMNIST integer labels.'); end
y=uint8(y);
end

function X=reshapeImages(raw,n)
if iscell(raw)
    pixels=zeros(784,n,'single');
    if numel(raw)~=n,error('FEMNIST x/y sample count mismatch.');end
    for k=1:n
        pixel=single(raw{k});
        if numel(pixel)~=784, error('FEMNIST image must have 784 pixels.');end
        pixels(:,k)=pixel(:);
    end
else
    raw=single(raw);
    if isequal(size(raw),[n,784])
        pixels=raw';
    elseif isequal(size(raw),[784,n])
        pixels=raw;
    elseif numel(raw)==n*784
        pixels=reshape(raw,784,n);
    else
        error('Unsupported LEAF FEMNIST x array shape.');
    end
end
if any(~isfinite(pixels(:))) || any(pixels(:)<0)
    error('Invalid FEMNIST pixels.');
end
if max(pixels(:))<=1+eps('single')
    pixels=255*pixels;
elseif max(pixels(:))>255
    error('FEMNIST pixel intensity exceeds 255.');
end
% LEAF uses row-major flattened 28x28 images. MATLAB column-major reshape
% needs transposition to retain image orientation.
X=permute(reshape(uint8(round(pixels)),28,28,1,n),[2,1,3,4]);
end
