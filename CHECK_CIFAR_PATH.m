%% CHECK_CIFAR_PATH.m
% Fast check of the supplied Windows path and the official CIFAR-10 MAT files.
clear; clc;

cfg = flcoppp_config("smoke");
rootPath = string(cfg.paths.cifarRoot);
candidates = [
    rootPath
    fullfile(rootPath,"cifar-10-batches-mat")
    fullfile(rootPath,"cifar-10-matlab","cifar-10-batches-mat")
];

found = "";
for i = 1:numel(candidates)
    required = [
        "data_batch_1.mat"
        "data_batch_2.mat"
        "data_batch_3.mat"
        "data_batch_4.mat"
        "data_batch_5.mat"
        "test_batch.mat"
    ];
    ok = true;
    for j = 1:numel(required)
        ok = ok && isfile(fullfile(candidates(i),required(j)));
    end
    if ok
        found = candidates(i);
        break;
    end
end

if strlength(found)==0
    fprintf(2,"CIFAR-10 was not found. Checked:\n%s\n",strjoin(candidates,newline));
    error("Fix cfg.paths.cifarRoot in flcoppp_config.m.");
end

fprintf("CIFAR-10 folder found:\n%s\n\n",found);
S = load(fullfile(found,"data_batch_1.mat"),"data","labels");
fprintf("data_batch_1: %d images, %d features per image\n",size(S.data,1),size(S.data,2));
fprintf("Label range: %d to %d\n",min(S.labels),max(S.labels));
fprintf("Deep Learning Toolbox available: %d\n",exist("dlnetwork","file")==2);
fprintf("Java/JVM available: %d\n",usejava("jvm"));
fprintf("\nSetup check passed.\n");
