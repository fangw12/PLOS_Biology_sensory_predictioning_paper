clear all
% load data
% vol_cond{subject,roi}: data in conditioning [run(1,2,3), voxel, cue type(B,D)]
% vol_probe_AC{subject,roi}: data in probe test [trials (1-8), voxel, cue type(A,C)]
% vol_probe_BD{subject,roi}: data in probe test [trials (1-8), voxel, cue type(B,D)]
% roi = {'medial OFC', 'lateral OFC', 'anterior HPC', 'posterior HPC'};
load('data.mat');

subj = 1:23;
runs = 1:3; % conditioning
rois = 1:4; % 1-2 OFC, 3-4 HPC
step = [40,40,20,20]; % 40 voxel increment for OFC, 20 for HPC
chance = 50;
addpath('/libsvm-3.14/');

nsubs = length(subj);
nrois = length(rois);
nruns = length(runs);


% create training and left-out subjects array
subj_minus_one = zeros(nsubs-1,nsubs-1);
for s = subj
    subj_minus_one(s,:) = setdiff(subj, s);
end
test_subj = subj;

%% compute decoding accuracies in all but left-out subjects
inference_decoding_accuracy = [];
paired_decoding_accuracy = [];

for subj_subset = subj
    
    s = 1;
    for sn = subj_minus_one(subj_subset,:)
        
        for m = rois
            nvox = size(vol_cond{sn,m},2);
            conj_included_size{m} = [step(m):step(m):fix(nvox/step(m))*step(m)]'; % number of included voxels
            
            % data in 3 conditioning runs
            data_train_tmp = vol_cond{sn,m}(:,:,1);
            data_train_tmp = [data_train_tmp; vol_cond{sn,m}(:,:,2)];
            
            % use t as a index of B-D difference
            [~,~,~,stats] = ttest(data_train_tmp(1:3,:),data_train_tmp(4:6,:));
            data_train_t = abs(stats.tstat);
            data_train_tmp_diff = sortrows([1:nvox'; data_train_t]',2)';
            
            %%%% compute decoding accuries for each included number of voxels
            voxel_index = 1;
            for w = conj_included_size{m}'
                lin_index_fs = data_train_tmp_diff(1,(nvox-w+1):end);
                
                %%%% training data in conditioning runs
                % B D
                data_train_fs = vol_cond{sn,m}(:,lin_index_fs,1);
                data_train_fs = [data_train_fs; vol_cond{sn,m}(:,lin_index_fs,2)];
                
                %%%% test data in probe test
                % A C
                data_testAC = vol_probe_AC{sn,m}(:,lin_index_fs,1);
                data_testAC = [data_testAC; vol_probe_AC{sn,m}(:,lin_index_fs,2)];
                
                % B D
                data_testBD = vol_probe_BD{sn,m}(:,lin_index_fs,1);
                data_testBD = [data_testBD; vol_probe_BD{sn,m}(:,lin_index_fs,2)];
                
                
                %%% train SVM
                training_label_vector = [ones(3,1); zeros(3,1)];
                testing_label_vector = [ones(8,1); zeros(8,1)];
                
                model = svmtrain(training_label_vector, data_train_fs,'-s 0 -t 0 -q');
                
                [~, accuracy_AC] = svmpredict(testing_label_vector,data_testAC, model,'-q');
                [~, accuracy_BD] = svmpredict(testing_label_vector,data_testBD, model,'-q');
                
                inference_decoding_accuracy{subj_subset,s,m}(voxel_index) = accuracy_AC(1)-chance;
                paired_decoding_accuracy{subj_subset,s, m}(voxel_index) = accuracy_BD(1)-chance;
                
                voxel_index = voxel_index + 1;
            end % w
        end % rois
        s = s + 1;
    end % subjects
    fprintf('feature selection set %02d\n',subj_subset);
end % subjects sets


%% compute optimal number of voxels for left-out subjects
for subj_subset = 1: length(subj)
    for m=rois
        smallest_size_inference(subj_subset,m) = [min(cellfun('size',inference_decoding_accuracy(subj_subset,1:22,m),2))];
        smallest_size_paired(subj_subset,m) = [min(cellfun('size',paired_decoding_accuracy(subj_subset,1:22,m),2))];
    end
end


included_voxel_inference = zeros((nsubs-1),nrois);
included_voxel_paired = zeros((nsubs-1),nrois);
for subj_subset = 1: length(subj)
    for m = rois
        inference_decoding_accuracy_sz_tmp = [];
        paired_decoding_accuracy_sz_tmp = [];
        for s = 1:length(subj)-1
            inference_decoding_accuracy_sz_tmp(s,:) = inference_decoding_accuracy{subj_subset,s,m}(1,1:smallest_size_inference(subj_subset,m));
            paired_decoding_accuracy_sz_tmp(s,:) = paired_decoding_accuracy{subj_subset,s,m}(1,1:smallest_size_paired(subj_subset,m));
        end
        
        [~,indx] = sort(mean(inference_decoding_accuracy_sz_tmp,1),2,'descend');
        included_voxel_inference(subj_subset,m) = indx(1)*step(m);
        
        [~,indx] = sort(mean(paired_decoding_accuracy_sz_tmp,1),2,'descend');
        included_voxel_paired(subj_subset,m) = indx(1)*step(m);
        
        if subj_subset==11 && m==4
            % subject 11 has fewer voxels than the optimal number based on all other subjects. So we take the next optimal number here that is small enough.
            included_voxel_inference(subj_subset,m) = indx(5)*step(m);
        end
        
    end
end


%% test on left out subjects
test_subj_paired_decoding_accuracy = zeros(nsubs,nrois);
test_subj_inference_decoding_accuracy = zeros(nsubs,nrois);

s = 1;
for sn = test_subj
    for m = rois
        nvox = size(vol_cond{sn,m},2);
        % data in 3 conditioning runs
        data_train_tmp = vol_cond{sn,m}(:,:,1);
        data_train_tmp = [data_train_tmp; vol_cond{sn,m}(:,:,2)];
        
        % use t as a index of B-D difference
        [~,~,~,stats] = ttest(data_train_tmp(1:3,:),data_train_tmp(4:6,:));
        data_train_t = abs(stats.tstat);
        data_train_tmp_diff = sortrows([1:nvox'; data_train_t]',2)';
        
        
        %%%%%%%%%%%%%%%%%% DECODING A vs C %%%%%%%%%%%%%%%%%%%%
        lin_index_fs = data_train_tmp_diff(1,(nvox-included_voxel_inference(s,m)+1):end);
        
        %%%% training data in conditioning runs
        % B D
        data_train_fs = vol_cond{sn,m}(:,lin_index_fs,1);
        data_train_fs = [data_train_fs; vol_cond{sn,m}(:,lin_index_fs,2)];
        
        %%%% test data in probe test
        % A C
        data_testAC = vol_probe_AC{sn,m}(:,lin_index_fs,1);
        data_testAC = [data_testAC; vol_probe_AC{sn,m}(:,lin_index_fs,2)];
        
        %%% SVM
        training_label_vector = [ones(3,1); zeros(3,1)];
        testing_label_vector = [ones(8,1); zeros(8,1)];
        model = svmtrain(training_label_vector, data_train_fs, '-s 0 -t 0 -q');
        [~, accuracy_AC] = svmpredict(testing_label_vector, data_testAC, model,'-q');
        test_subj_inference_decoding_accuracy(s,m) = accuracy_AC(1);
        
        %%% compute empirical chance
        accuracy_AC_shuffle_all = [];
        for emp = 1:10000
            testing_label_vector_shuffle = testing_label_vector(randperm(length(testing_label_vector)));
            [~, accuracy_AC_shuffle] = svmpredict(testing_label_vector_shuffle, data_testAC,model, '-q');
            accuracy_AC_shuffle_all(emp) = accuracy_AC_shuffle(1);
        end
        test_subj_inference_decoding_accuracy_shuffle(s,m) = mean(accuracy_AC_shuffle_all);
        
        
        %%%%%%%%%%%%%%%%%% DECODING B vs D %%%%%%%%%%%%%%%%%%%%%%%
        lin_index_fs = data_train_tmp_diff(1,(nvox-included_voxel_paired(s,m)+1):end);
        
        %%%% training data in conditioning runs
        % B D
        data_train_fs = vol_cond{sn,m}(:,lin_index_fs,1);
        data_train_fs = [data_train_fs; vol_cond{sn,m}(:,lin_index_fs,2)];
        
        %%%% test data in probe test
        % B D
        data_testBD = vol_probe_BD{sn,m}(:,lin_index_fs,1);
        data_testBD = [data_testBD; vol_probe_BD{sn,m}(:,lin_index_fs,2)];
        
        %%% SVM
        training_label_vector = [ones(3,1); zeros(3,1)];
        testing_label_vector = [ones(8,1); zeros(8,1)];
        model = svmtrain(training_label_vector, data_train_fs, '-s 0 -t 0 -q');
        [~, accuracy_BD] = svmpredict(testing_label_vector, data_testBD, model, '-q');
        test_subj_paired_decoding_accuracy(s,m) = accuracy_BD(1);
        
        
        %%% compute empirical chance
        accuracy_BD_shuffle_all = [];
        for emp = 1:10000
            testing_label_vector_shuffle = testing_label_vector(randperm(length(testing_label_vector)));
            [~, accuracy_BD_shuffle] = svmpredict(testing_label_vector_shuffle, data_testBD(:,:),model,'-q');
            accuracy_BD_shuffle_all(emp) = accuracy_BD_shuffle(1);
        end
        test_subj_paired_decoding_accuracy_shuffle(s,m) = mean(accuracy_BD_shuffle_all);
        
    end
    fprintf('decoding subejct %02d completed \n',sn);
    s = s + 1;
end

%% test decoding accuracies against empirical chance
[h,p,ci,stats] = ttest(test_subj_paired_decoding_accuracy(:,1),test_subj_paired_decoding_accuracy_shuffle(:,1),'Tail','right') % medial OFC
[h,p,ci,stats] = ttest(test_subj_paired_decoding_accuracy(:,2),test_subj_paired_decoding_accuracy_shuffle(:,2),'Tail','right') % lateral OFC
[h,p,ci,stats] = ttest(test_subj_paired_decoding_accuracy(:,3),test_subj_paired_decoding_accuracy_shuffle(:,3),'Tail','right') % anterior HPC
[h,p,ci,stats] = ttest(test_subj_paired_decoding_accuracy(:,4),test_subj_paired_decoding_accuracy_shuffle(:,4),'Tail','right') % posterior HPC

[h,p,ci,stats] = ttest(test_subj_inference_decoding_accuracy(:,1),test_subj_inference_decoding_accuracy_shuffle(:,1),'Tail','right') % medial OFC
[h,p,ci,stats] = ttest(test_subj_inference_decoding_accuracy(:,2),test_subj_inference_decoding_accuracy_shuffle(:,2),'Tail','right') % lateral OFC
[h,p,ci,stats] = ttest(test_subj_inference_decoding_accuracy(:,3),test_subj_inference_decoding_accuracy_shuffle(:,3),'Tail','right') % anterior HPC
[h,p,ci,stats] = ttest(test_subj_inference_decoding_accuracy(:,4),test_subj_inference_decoding_accuracy_shuffle(:,4),'Tail','right') % posterior HPC


