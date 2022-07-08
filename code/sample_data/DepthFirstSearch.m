function [dfs_path, NVisited, ComponentStartNode, NComponents, ComponentStartInDfsPath] = DepthFirstSearch(adjacency,startNode)
% DEPTHFIRSTSEARCH Implementation of the standart depth-first search

%%% DESCRIPTION:
% This function performs a depth-first search throug the given graph using
% the adjacency matrix and a the start node as input. Returns a dfs-path.

%%% INPUT: 
% adjacency = adjacency matrix
% startNode = number of start node for dfs-search

%%% OUTPUT:
% dfs_path = path of the dfs-search (order of the visited nodes)
% NVisited = number of visited nodes
% ComponentStartNode = start node of the identified components
% NComponets = number of identified components
% ComponentStartInDfstree = index in dfs-path where component starts

%% Node information
NNodes = length(adjacency);
Nodes = 1:NNodes;
UnvisitedNodes = Nodes;
NComponents = 0;
visited = [];
dfs_path = [];
ComponentStartNode = [];
ComponentStartInDfsPath = [];

%% loop through unvisited nodes
% as long as there are unvisited nodes do depth search until all nodes have 
% been visited. If the graph is not connected dfs has to be started 
% multiple times.
while ~isempty(UnvisitedNodes)

    %% Counting Components
    NComponents = NComponents + 1;
    ComponentStartNode = [ComponentStartNode, startNode];
    
    %% initializing dfs
    dfs_path = [dfs_path, startNode];
    CurrentNode = startNode;
    ComponentStartInDfsPath = [ComponentStartInDfsPath,length(dfs_path)];
    Neighbors = Nodes(adjacency(CurrentNode,:) == 1);
    
    %% mark start node as visited
    visited = [visited, startNode];
    IsVisited = ismember(Neighbors, visited);
    UnvisitedNeighbors = Neighbors(~IsVisited);

    %% Actual depth-first search
    % loop until backtracking comes to root and doesn't find further neighbors
    while CurrentNode ~= startNode || ~isempty(UnvisitedNeighbors)
        
        % backtracking if no neigbors and not at component root
        if isempty(UnvisitedNeighbors) && CurrentNode ~= startNode

            % backtracking: goes back to previous node
            InStack = find(dfs_path == CurrentNode);
            FirstInStack = InStack(1);
            CurrentNode = dfs_path(FirstInStack-1);
            
            % update neighbor information
            Neighbors = Nodes(adjacency(CurrentNode,:) == 1);
            IsVisited = ismember(Neighbors, visited);
            UnvisitedNeighbors = Neighbors(~IsVisited);
            
            % if backtracking finished save backtrack node
            if ~isempty(UnvisitedNeighbors)
                dfs_path = [dfs_path, CurrentNode];
            elseif isempty(UnvisitedNeighbors) && length(find(dfs_path == CurrentNode)) > 1
                dfs_path = [dfs_path, CurrentNode];
            end
        
        else

            % visit first unvisited node in adjacency list
            CurrentNode = UnvisitedNeighbors(1);
            visited = [visited, CurrentNode];
            dfs_path = [dfs_path, CurrentNode];

            % update information on vistited nodes
            Neighbors = Nodes(adjacency(CurrentNode,:) == 1);
            IsVisited = ismember(Neighbors, visited);
            UnvisitedNeighbors = Neighbors(~IsVisited);
            
        end

    end

% add new start node to dfs path component
dfs_path = [dfs_path, startNode];

% update Unvisited Nodes
NodeVisited  = ismember(Nodes, visited);
UnvisitedNodes = Nodes(NodeVisited == 0);

% update start node for iterating dfs through new component
if ~isempty(UnvisitedNodes)
    startNode = UnvisitedNodes(1);
end
    
end

% number of visited nodes 
NVisited = length(visited);

% Complete Component range information
NComponents = length(ComponentStartNode);

end