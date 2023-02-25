%% PRIME SVM Tutorial
% using our lab's RFT PPE data set (first level models made by Qi)
% this workshop requires SPM12 and CanlabCore
% analysis workbook: https://docs.google.com/document/d/1TBmq2Rnf4YDwEOCLpINY7NtsIGQZbpfcw9FYrZXN7kU/edit

%% Set Up
% Navigate to your analysis folder
analysis_dir = '/Users/marianne/Desktop/projects/SVM_Workshop'; % replace this with your filepath
cd(analysis_dir);
% addpath(genpath('/Users/mariannereddan/coderepos/')); % addpath to the necessary repos: CanlabCore & SPM
addpath(genpath('/Users/marianne/code'));

%% What is your question?
% ---- Can we predict cannabis use from reward-related brain activity?
%
% What is your hypothesis?
% H1
% ---- Chronic cannabis use alters endocannabinoid receptor density and activity
% in brain regions like the nucleus accumbens (NAc). Therefore, we expect that
% reward-related brain activity, which is largely driven by signalling  in
% the NAc will be different in chronic THC users versus sober controls.
% H1: Brain activity during reward will be different for THC users vs HC
%
% H2
% ---- Reward tends to involve these brain regions... 
% see: https://neurosynth.org/analyses/terms/reward/
% PFC, OFC, Striatum, VTA, substantia nigra, PCC
% ---- THC impacts these brain regions: ....
% see: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3721267/
% Cannabinoid CB1 receptors are found at very high concentrations in the cerebellum
% chronic THC interfered with hippocampal LTP via activation of CB1 receptors 
% Dorsal & Ventral Striatum
% maybe primary motor areas? (The dorsal striatum plays an integral role in
% motor behavior, and the ability of THC to induce hypolocomotion and ataxia is thought to result from activation of CB1 receptors in this and other brain area)
% OFC
% Therefore, we expect reward-related brain activity in the ROIs listed below to
% drive the differences in brain activity between these two groups.
% H2: The most important brain regions for this prediction will be: PFC, OFC
% Dorsal & Ventral Striatum
%
% Listing your ROIs ahead of time justifies 'feature selection' later on
%
% Set up the analysis to answer your question
%
% What are you predicting (Y)? 
%  ---- THC-use, binary yes/no
load sub_id_list_with_THC_dat % this is a cell matrix where the first column is THC use id (1 = user, -1 = sober) and the second column is a list of subject IDs that match the THC data
thc_label = cell2mat(sub_id_list_with_THC_dat(:,1)); % put the THC labels into a new vector, that is not a cell

% What are your predicting this from (X)?
%  ---- Whole brain activity during reward attainment (during the RFT task)
%  Note, in Qi's SPM contrast data reward expectation is Contrast 1, and the reward attainment is Contrast 27
%
%  generate a file list of the brain data we want to analyze - Contrast 27 (Reward Attainment) - con_0027.nii
dat_list = filenames('data/1st_level/*/model_01/con_0027.nii');
% load the files into an fmri_data object
dat = fmri_data(dat_list);
dat.Y = thc_label; % add the label data to the fmri data object
%% Run the analysis
% run linear svm, c = 1, with leave-5-subj-out cross validation
use_spider;
[cverr, stats, optout] = predict(dat, 'algorithm_name', 'cv_svm', 'nfolds', 3, 'error_type', 'mcr');
% disp(sprintf('CV ACC: %d %',1-cverr));
% print the CV- training accuracy
fprintf('map has a %f accuracy',(1-stats.cverr));
% approx 52% accuracy with MCR not better than chance

% plot the receiver operating characteristic
ROC = roc_plot(stats.dist_from_hyperplane_xval,logical(stats.Y>0),'plothistograms');
saveas(gcf, 'ROCplot.png');
% ROC_PLOT Output: Single-interval, Optimal overall accuracy
% Threshold:	0.26	Sens:	  9% CI(0%-24%)	Spec:	100% CI(100%-100%)	PPV:	100% CI(100%-100%)	Nonparametric AUC:	0.47	Parametric d_a:	-0.10	  Accuracy:	 62% +- 6.7% (SE), P = 0.678537


% display the SVM weight map - unthresholded
orthviews(stats.weight_obj)

%% bootstrap the prediction to get significance estimates for the SVM weights
% should bootstrap ~10,000 samples... doing 500 for computational ease
[cverr, stats, optout] = predict(dat, 'algorithm_name', 'cv_svm', 'nfolds', 3,'bootsamples', 500);

% plot the thresholded map
obj=statistic_image();
obj.dat=stats.WTS.wZ'; % these are the weight values
obj.p=stats.WTS.wP'; % these are the p values for the significance of each weight
obj.volInfo=stats.weight_obj.volInfo; % carries information for plotting

%thresholds the image with FDR correction q < 0.05, cluster size k =1
obj = threshold(obj, .05, 'fdr', 'k', 1, 'mask', which('gray_matter_mask.img'))
cl=region(obj); % change image object format for plotting
table=cluster_table(cl, 0, 0,'writefile','RewardAttainment_THCSig_FDR05_K0_clusttable');
o2=canlab_results_fmridisplay();
o2=addblobs(o2,cl,'splitcolor', {[0 0 1] [.3 0 .8] [.8 .3 0] [1 1 0]})
obj.fullpath=['RewardAttainment_TCHPredictivePattern.nii'];
write(obj,'thresh','keepdt');
saveas(gcf, 'SVMweightmap_FDR05_K0.png');


% Now you try
% look at Contrast_SPM.xlsx and try to predict THC use from a different set
% of contrast maps that makes sense to you
% or
% don't predict THC, predict something like reward uncertainty from
% certainty within all subjects -- for this you will need to drop WHOLE
% subjects from the Cross Validation (ask M to help you)
