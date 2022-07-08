function runDataBaseGeneration(NFullyLabeled,NSingleLabeled,NSamples)
% RUNDATABASEGENERATION Samples a database of appropriate size

% DESCRIPTION:
% The script generates a data base of desired size. Therefore, different
% production parameter combinations are sampled and for each combination 
% multiple (number = NSamples) fiber graphs are generated. Then either
% every graph (fully labeled) or just a single graph (single labled) is 
% labeled with a stress-strain curve parametrization

% INPUT:
% NFullyLabeled - desired number of fully labeled data sets
% NSingleLabeled - desired number of single labeled data sets

% OUTPUT:
% exports graphml and csv file to the respecitve data folders

% initial output
fprintf('Generate %i fully labeled data sets and %i sinlge labeled data sets \n', NFullyLabeled, NSingleLabeled)

% set random seed
rng('default')

% bounds spanning the 4-parametric process class
alpha_solid_bounds = [1000,1500]; 
sigma_ramp_bounds = [1,5];
sigma_sde_bounds = [1,5];
kappa_bounds = [2.8e-2,3.0e-2];

% generate labeled (j==1) and unlabeled (j==2) data sets
for j = 1:2

    % set number of labeled/unlabeled samples
    if j == 1
        NSets = NFullyLabeled;
        fprintf('Start Generating fully labeled Sets \n')
        randseedoffset = 0;
    else
        NSets = NSingleLabeled;
        fprintf('Start Generating single labeled Sets \n')
        randseedoffset = NFullyLabeled*NSamples;
    end

    % sample labeled/unlabeled data from 4-parametric process class
    for i = 1:NSets

        % sample random production parameter combination
        alpha_solid =  alpha_solid_bounds(1) + rand() * (alpha_solid_bounds(2) - alpha_solid_bounds(1));
        sigma_ramp =  sigma_ramp_bounds(1) + rand() * (sigma_ramp_bounds(2) - sigma_ramp_bounds(1)); 
        sigma_sde = sigma_sde_bounds(1) + rand() * (sigma_sde_bounds(2) - sigma_sde_bounds(1));
        kappa = kappa_bounds(1) + rand() * (kappa_bounds(2) - kappa_bounds(1));
        
        % parallelization of data set generation 
        parfor k = 1:NSamples

            % Set random seed for parfor loop 
            fprintf('Sample: %i / %i \n', randseedoffset+(i-1)*NSamples+k,(NFullyLabeled+NSingleLabeled)*NSamples)
            rng(randseedoffset+(i-1)*NSamples+k, 'twister');

            % generate fiber structure (not adhered)
            [eta,startFib,endFib,startType,endType,NFib_sld,NFib_adh] = FiberWebGenerator(alpha_solid, sigma_ramp, sigma_sde);

            % virtual bonding (yields adhered fiber structure topology)
            [A,B,node_type,node_coordinate,FiberOnEdge] = GraphGenerator(eta, startFib, endFib, startType, endType, NFib_sld, NFib_adh, kappa);

            % data reduction (graph cleansing)
            if ~isempty(A) 
                % if connected from top to bottom
                [A_red, B_red, node_type_red, node_coordinate_red, FiberOnEdge_red] = DataReduction(A, B, node_type, node_coordinate, FiberOnEdge);
            else
                % else return empty graph
                A_red = 0; B_red = 0; node_type_red = 0; node_coordinate_red = 0; FiberOnEdge_red = 0;
            end

            %% label data and export results
            if j == 1 || (j == 2 && k == 1)

                % precomputation (assembles differential equation system)
                TestVolume = TestVolumePreparation(A_red, B_red, node_type_red, node_coordinate_red, FiberOnEdge_red);

                % perform tensile strength simulation (stress-strain curve computation)
                Output = 0;
                targetStrain = 0.5;
                regFriction = 1e-7;
                regMaterialLaw = 1e-4;
                [strain,stress,~,~] = TensileStrengthSimulation(TestVolume, targetStrain, regFriction, regMaterialLaw, Output);

                % export graph (graphml) and tensile strength simulation results (csv)
                subfoldername = 'input_data_labelled/';
                SaveResults(subfoldername, alpha_solid, sigma_ramp, sigma_sde, kappa, B_red, FiberOnEdge_red, node_type_red, node_coordinate_red,...
                            k, stress, strain);

            %% unlabeled data gets exported directly
            else

                % export graph (graphml)
                subfoldername = 'input_data_graphonly/';
                SaveResults(subfoldername, alpha_solid, sigma_ramp, sigma_sde, kappa, B_red, FiberOnEdge_red, node_type_red, node_coordinate_red, k);

            end
        
        end

    end

end

% final output
fprintf('Finished Dataset Generation \n')

end