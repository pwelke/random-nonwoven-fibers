function [adjacency, incidency, node_type, node_coordinate, fiber_on_edge] = DataReduction(adjacency, incidency, node_type, node_coordinate, fiber_on_edge)
% DATAREDUCTION Removes subgraphs not contributing to the tensile behavior

% DESCRIPTION:
% Parts of the fiber structure that do not contribute to the tensile
% strength behavior are removed. This includes uninvolved components, loose
% subgraphs as well as merging fiber connections that are linked over a
% node of degree 2 (where each edge indicent to the node represents one
% fiber connection only).

% INPUT: 
% adjacency = adjaceny of graph representing adhered fiber web topology
% incidency = incidency of graph representing adhered fiber web topology
% node_type = indicates if node is inner node (0) or lateral face node (1-6)
% node_coordinate = spartial coordinate of nodes
% fiber_on_edge = assigns fibers and their lengths to the edges

% OUTPUT
% adjacency = reduced adjaceny of graph representing adhered fiber web topology
% incidency = reduced incidency of graph representing adhered fiber web topology
% node_type = reduced indicates if node is inner node (0) or lateral face node (1-6)
% node_coordinate = reduced spartial coordinate of nodes
% fiber_on_edge = reduced assigns fibers and their lengths to the edges


%% 
% reduction 1: (remove degree zero nodes and free moving degree one nodes, 
% doing this is cheap and reduces the up coming effort)

% get initial structural information
[NNode,NEdge] = size(incidency);
Nodes = (1:NNode)';
Edges = (1:NEdge)';
IsFixedNode = (node_type > 4);

% loop until no free moving degree 1 node can be found anymore
NNodeOld = NNode;
NNodeNew = 0;
while NNodeOld > NNodeNew
    
    % node 0 and free moving node 1 node identification
    NodeRanks = sum(adjacency,2); 
    RankZeroNodes = Nodes(NodeRanks == 0);
    IsFreeMovingRankOne = logical((NodeRanks == 1).*(~IsFixedNode));
    FreeMovingRankOneNodes = Nodes(IsFreeMovingRankOne);
    RemovableNodes = [RankZeroNodes,FreeMovingRankOneNodes];
    IsRemovableNode = ismember(Nodes, RemovableNodes)';
    IsRemovableEdge = (sum(abs(incidency(IsRemovableNode,:)),1) >= 1)';

    % adjacency and incidence manipulation
    [NNodeOld, ~] = size(incidency);
    adjacency = adjacency(~IsRemovableNode, ~IsRemovableNode);
    incidency = incidency(~IsRemovableNode, ~IsRemovableEdge);
    [NNodeNew, NEdgeNew] = size(incidency);

    % node coordinates, node types and fiber on edge matrix manipulation
    node_coordinate = node_coordinate(~IsRemovableNode,:);
    node_type = node_type(~IsRemovableNode);
    IsFixedNode = IsFixedNode(~IsRemovableNode);
    IsRemainingFiber = ismember(fiber_on_edge(:,2), Edges(~IsRemovableEdge));
    fiber_on_edge = fiber_on_edge(IsRemainingFiber, :);
    [NFibNew, ~] = size(fiber_on_edge);

    % reset fiber numbering and replace edge numbering in fiber on edge matrix
    fiber_on_edge(:,1) = (1:NFibNew)';
    OldEdgeNumber = Edges(~IsRemovableEdge);
    NewEdgeNumber = (1:NEdgeNew)';
    for CurrentFiber = 1:NFibNew
        OldEdge = fiber_on_edge(CurrentFiber,2);
        fiber_on_edge(CurrentFiber,2) = NewEdgeNumber(OldEdgeNumber == OldEdge);
    end 
    
    % reset edge and node numbering
    Nodes = 1:NNodeNew;
    Edges = 1:NEdgeNew;
end


%% 
% reduction 2: (depth first search to get the dfs-path)

% set initial dfs-node and start search
StartNode = 1;
[dfs_path, ~, ComponentStartNode, NComponents] = DepthFirstSearch(adjacency,StartNode);


%%
% reduction 3: (identification of uninvolved components)

% create array indecating the deletable Components
RemovableComponent = false(1,length(dfs_path));

% create matrix that is false if corresponding part of dfs path is removable
dfs_path_remaining = true(1,length(dfs_path));
NBottomTopConnections = 0;
IsUpperBoundaryNode = (node_type == 5);
IsLowerBoundaryNode = (node_type == 6);
UpperBoundaryNodes = Nodes(IsUpperBoundaryNode);
LowerBoundaryNodes = Nodes(IsLowerBoundaryNode);
FixedNodes = Nodes(IsFixedNode);

% loop thorugh components
for CurrentComponent = 1:NComponents
    
    % determine i-th component in dfs path 
    InStack = find(dfs_path == ComponentStartNode(CurrentComponent));
    ComponentStartInStack = InStack(1);
    ComponentEndInStack = InStack(end);
    Component = dfs_path(ComponentStartInStack:ComponentEndInStack);

    % check wheather the component has a relevant connection (bottom-top)
    BottomTopConnection = logical(any(ismember(Component, UpperBoundaryNodes)) * any(ismember(Component, LowerBoundaryNodes)));
    % if there is no bottom-top connection then component is removable
    if ~BottomTopConnection
        % mark component as removable
        RemovableComponent(ComponentStartInStack:ComponentEndInStack) = 1;
        % mark the position of the dfs_path as removable
        dfs_path_remaining(ComponentStartInStack:ComponentEndInStack) = 0;
    else
        NBottomTopConnections = NBottomTopConnections + 1;
    end
    
end

% all nodes within a removable component can be deleted
RemovableComponents = unique(dfs_path(RemovableComponent));
% truncation of the original dfs_path
dfs_path = dfs_path(dfs_path_remaining);


%%
% reduction 4: (identification of loose subgraphs)
NodeRanks = sum(adjacency,2);                    % rank computation
PossibleCutvertices = Nodes(NodeRanks >= 2);     % cutvertices have at least 2 neighbors
RemovableBranchNodes = [];

% find cutvertices since they might indicate removable subgraphs
for BranchRoot = PossibleCutvertices
    
    % find node in dfs stack
    NodeInStack = find(dfs_path == BranchRoot);
    
    % every time node occures in dfs_path new brach emerges. Count nodes
    % in stack to determine the outgoing branches
    NBranches = length(NodeInStack)-1;
    
    % loop through branch roots
    for CurrentNode = 1:NBranches
        
        % identify branch
        BranchStart = NodeInStack(CurrentNode);
        BranchEnd = NodeInStack(CurrentNode+1);
        BranchNodes = unique(dfs_path(BranchStart:BranchEnd));
        InnerBranchNodes = BranchNodes(BranchNodes ~= BranchRoot);
        InnerBranchAdjacencyRows = adjacency(InnerBranchNodes,:);
        InnerBranchNeighbors = unique([Nodes(sum(InnerBranchAdjacencyRows,1) > 0), BranchNodes]);
        
        % if the branch nodes are the same as its inner neigbors and none of the
        % branch nodes is fixed, then remove the branch. The condition that
        % branch nodes are equal to its inner node neigbors is equivalent
        % to the branch being a loose subgraph.
        if length(BranchNodes) == length(InnerBranchNeighbors) && all(ismember(InnerBranchNeighbors, BranchNodes)) && ~any(ismember(FixedNodes, InnerBranchNodes))
           RemovableBranchNodes = [RemovableBranchNodes, InnerBranchNodes];
        end
        
    end
    
end

% removable nodes resulting from loose graph removal
RemovableBranchNodes = unique(RemovableBranchNodes);


%%
% reduction 5: (actual removal of uninvolved components and loose subgraphs)

% actual removal of removable nodes detected in step 1 and 2
RemovableNodes = unique([RemovableBranchNodes, RemovableComponents]);
IsRemovableNode = ismember(Nodes, RemovableNodes)';
IsRemovableEdge = (sum(abs(incidency(IsRemovableNode,:)),1) >= 1)';

% adjacency and incidence manipulation
adjacency = adjacency(~IsRemovableNode, ~IsRemovableNode);
incidency = incidency(~IsRemovableNode, ~IsRemovableEdge);
[NNodeNew, NEdgeNew] = size(incidency);

% node coordinates, node types and fiber on edge matrix manipulation
node_coordinate = node_coordinate(~IsRemovableNode,:);
node_type = node_type(~IsRemovableNode);
IsFixedNode = IsFixedNode(~IsRemovableNode);
IsRemainingFiber = ismember(fiber_on_edge(:,2), Edges(~IsRemovableEdge));
fiber_on_edge = fiber_on_edge(IsRemainingFiber, :);
[NFibNew, ~] = size(fiber_on_edge);

% reset fiber numbering and replace edge numbering in FiberOnEdge matrix
fiber_on_edge(:,1) = (1:NFibNew)';
OldEdgeNumber = Edges(~IsRemovableEdge);
NewEdgeNumber = (1:NEdgeNew)';
for CurrentFiber = 1:NFibNew
    OldEdge = fiber_on_edge(CurrentFiber,2);
    fiber_on_edge(CurrentFiber,2) = NewEdgeNumber(OldEdgeNumber == OldEdge);
end


%%
% reduction 6: (merging directly linked fiber connections)

% identify free moving degree-2-nodes
CurrentNode = 1; 
NNodes = NNodeNew;
NEdges = NEdgeNew;

% loop through free moving degree-2-nodes and check if they connect a
% mergable fiber connection pair. if so, mark for removal
while CurrentNode <= NNodes

     % Indefification of free moving degree-2-nodes (FDT-Nodes)
     IsFDTNode = (sum(adjacency(CurrentNode,:)) == 2) && ~IsFixedNode(CurrentNode);

     % Compute number of incident fibers
     Nodes = 1:NNodes;
     Edges = 1:NEdges;
     IncidentEdges = Edges(logical(incidency(CurrentNode,:)));
     NIncidentFibers = length(fiber_on_edge(ismember(fiber_on_edge(:,2),IncidentEdges),1));

     % if node links two fiber connections
     if NIncidentFibers == 2 && IsFDTNode
         
        %%% Neighbor detection
        IsNodeNeighbor = logical(adjacency(CurrentNode,:));
        Neighbors = Nodes(IsNodeNeighbor);

        %%% Adjust Adjacency Matrix
        % Add new neigbor relation
        adjacency(Neighbors(1),Neighbors(2)) = 1;
        adjacency(Neighbors(2),Neighbors(1)) = 1;
        % Remove linking node
        IsDeletableNode = (Nodes == CurrentNode);
        adjacency = adjacency(~IsDeletableNode, ~IsDeletableNode);

        %%% Adjust Incidence Matrix
        % replace first incident with new connection
        incidency(:,IncidentEdges(1)) = zeros(length(Nodes),1);
        incidency(Neighbors(1),IncidentEdges(1)) = 1;
        incidency(Neighbors(2),IncidentEdges(1)) = -1;
        % delete second incident edge
        IsDeleatableEdge = (Edges == IncidentEdges(2));
        incidency = incidency(~IsDeletableNode, ~IsDeleatableEdge);

        %%% Adjust node information
        node_coordinate = node_coordinate(~IsDeletableNode,:);
        node_type = node_type(~IsDeletableNode); 
        IsFixedNode = IsFixedNode(~IsDeletableNode);

        %%% Adjust FiberOnEdge Matrix
        % merge the fiber lengths
        accumlength = fiber_on_edge(fiber_on_edge(:,2) == IncidentEdges(1),3) + fiber_on_edge(fiber_on_edge(:,2) == IncidentEdges(2),3);
        fiber_on_edge(fiber_on_edge(:,2) == IncidentEdges(1),3) = accumlength;
        fiber_on_edge = fiber_on_edge(fiber_on_edge(:,2) ~= IncidentEdges(2),:);
        fiber_on_edge(:,2) = fiber_on_edge(:,2) .* (fiber_on_edge(:,2) < IncidentEdges(2)) + (fiber_on_edge(:,2)-1).* (fiber_on_edge(:,2) > IncidentEdges(2));
        fiber_on_edge(:,1) = (1:length(fiber_on_edge))';
       
        % update node and edge numbers 
        [NNodes,NEdges] = size(incidency);
        
     else
         
        %%% update current node number
        CurrentNode = CurrentNode + 1;
        
     end
    
end

% Update graph size
fprintf('Graph size after reduction: nodes %i, edges %i \n',NNodes,NEdges)

end