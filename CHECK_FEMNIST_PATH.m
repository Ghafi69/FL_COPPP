%% CHECK_FEMNIST_PATH.m -- verify LEAF JSON shards are accessible
clear; clc;
c=flcoppp_apply_femnist_config(flcoppp_config("paper"));
root=string(c.paths.femnistRoot);
choices=[root;fullfile(root,'data');fullfile(root,'femnist','data')];
found=false;
for i=1:numel(choices)
    if ~isempty(dir(fullfile(choices(i),'train','*.json'))) && ...
       ~isempty(dir(fullfile(choices(i),'test','*.json')))
        fprintf('LEAF FEMNIST JSON found: %s\n',choices(i));found=true;break;
    end
end
assert(found,'Cannot locate LEAF FEMNIST train/test JSON shards.');
