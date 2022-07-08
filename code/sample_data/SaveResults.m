function [fname_graphml,fname_csv] = SaveResultsDataSetGen(subfoldername, alpha_solid, sigma_ramp, sigma_sde, kappa, incidency, fiber_on_edge, node_type, node_coordinate, number, stress, strain)
% SAVERESULTSDATASETGEN Generates full dataset for training

% DESCRIPTION:
% formats and saves graph to graphml file and exports the simulates
% stress-strain curves as csv file.

% INPUT:
% alpha_solid, sigma_ramp, sigma_sde, kappa - production parameters
% incidency, fiber_on_edge, node_type, node_coordinate - graph infromation
% stress, strain - stress strain curve

% OUTPUT:
% exports graphml and csv file to the respective data folders


%% 
% file naming

% formatting
formatSpec_sigma = '%.4f';
formatSpec_kappa = '%.6f';

% get date
date_str = num2str(yyyymmdd(datetime));
date_str = [date_str(1:4),'_',date_str(5:6),'_',date_str(7:8)];

 % naming rules
str_sld = num2str(round(alpha_solid));
str_sigma_ramp = strrep(num2str(sigma_ramp,formatSpec_sigma),'.','p');
str_sigma_ramp = str_sigma_ramp(1:5);
str_sigma_sde = strrep(num2str(sigma_sde,formatSpec_sigma),'.','p');
str_sigma_sde = str_sigma_sde(1:5);
str_kappa = strrep(num2str(kappa,formatSpec_kappa),'.','p');
str_kappa = str_kappa(1:8);
str_sample = num2str(number);

% file naming
fname_graphml = ['/results/',subfoldername,date_str,'_Sld',str_sld,'_SigRamp',str_sigma_ramp,'_SigSde',str_sigma_sde,'_Kappa',str_kappa,'_N',str_sample,'_Microstructure.graphml'];
fname_csv = ['/results/',subfoldername,date_str,'_Sld',str_sld,'_SigRamp',str_sigma_ramp,'_SigSde',str_sigma_sde,'_Kappa',str_kappa,'_N',str_sample,'_StressStrainCurve.csv'];


%%
% export graph to graphml format

if ~isempty(incidency)
    
    % Graph Information
    [NNodes,~] = size(incidency);
    [NFibers,~] = size(fiber_on_edge);
    Nodes = 1:NNodes;

    % Save As .gml
    fid=fopen(fname_graphml,'w');
    % XML verison
    fprintf(fid,'%s\n','<?xml version="1.0" encoding="UTF-8"?>');
    % Begin GraphML
    fprintf(fid,'%s\n','<graphml xmlns="http://graphml.graphdrawing.org/xmlns"  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"  xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">');

    % Node Attributes
    fprintf(fid,'%s\n','  <key id="d1" for="node" attr.name="NodeType" attr.type="integer"/>');
    fprintf(fid,'%s\n','  <key id="d2" for="node" attr.name="x_Val" attr.type="double"/>');
    fprintf(fid,'%s\n','  <key id="d3" for="node" attr.name="y_Val" attr.type="double"/>');
    fprintf(fid,'%s\n','  <key id="d4" for="node" attr.name="z_Val" attr.type="double"/>');

    % Edge Attributes
    fprintf(fid,'%s\n','  <key id="d5" for="edge" attr.name="Length" attr.type="double"/>');

     % Begin Graph
    fprintf(fid,'%s\n','  <graph id="G" edgedefault="undirected">');

    % Nodes
    for ii = 1:NNodes
     fprintf(fid,'%s\n', ['      <node id="n', num2str(ii), '">']);
     fprintf(fid,'%s\n', ['        <data key="d1">', num2str(node_type(ii)), '</data>']);
     fprintf(fid,'%s\n', ['        <data key="d2">', num2str(node_coordinate(ii,1)), '</data>']);
     fprintf(fid,'%s\n', ['        <data key="d3">', num2str(node_coordinate(ii,2)), '</data>']);
     fprintf(fid,'%s\n', ['        <data key="d4">', num2str(node_coordinate(ii,3)), '</data>']);
     fprintf(fid,'%s\n', ['      </node>']);
    end

    % Edges
    for jj = 1:NFibers
     IncidentNodes = Nodes(incidency(:,fiber_on_edge(jj,2))~=0);
     fprintf(fid,'%s\n', ['      <edge id="e', num2str(jj), '" source="n', num2str(IncidentNodes(1)),'" target="n', num2str(IncidentNodes(2)),'">']);
     fprintf(fid,'%s\n', ['        <data key="d5">', num2str(fiber_on_edge(jj,3)), '</data>']);
     fprintf(fid,'%s\n', ['      </edge>']);
    end

    fprintf(fid,'%s\n','  </graph>');
    % End Graph

    fprintf(fid,'%s','</graphml>');
    % End GraphML

    fclose(fid);
    
end


%%
% export stress-strain curve to csv (if they are provided)
if nargin == 12
    ColNames = {'Strain','Stress'}; ExportData = [strain',stress'];
    T = array2table(ExportData,'VariableNames',ColNames);
    writetable(T,fname_csv)
end

% display output
fprintf('Exportet data sucessfully \n')
end