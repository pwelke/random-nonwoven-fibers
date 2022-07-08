function [adjacency,incidency,node_type,node_coordinate,fiber_on_edge] = GraphGenerator(eta, startFib, endFib, startType, endType, NFib_sld, NFib_adh, kappa)
% GRAPHGENERATOR Assembles graph representing the adhered fiber web

%%% DESCRIPTION:
% Create adhesive joints between fiber pairs if they posses a spartial
% discretization point pair whos distance falls below contact threshold EPS
% and if at least one of them is of bi-component type. 

% INPUT: 
% eta = three-dimensional spartial discretization points of all fibers
% startFib = indicate start points of the individual fibers in eta
% endFib = indicate end points of the individual fibers in eta
% startType = indicates where fiber starts (in test volume, or outer face) 
% endType = indicates where fiber end (in test volume, or outer face)
% NFib_sld = number of sampled solid fibers
% NFib_sld = number of sampled bi-component fibers
% kappa = contact threshold for virtual bonding

% OUTPUT
% adjacency = adjaceny of graph representing adhered fiber web topology
% incidency = incidency of graph representing adhered fiber web topology
% node_type = indicates if node is inner node (0) or lateral face node (1-6)
% node_coordinate = spartial coordinate of nodes
% fiber_on_edge = assigns fibers and their lengths to the edges



%% pair graph generation
% First all contact point are identified. Each contact point represents a 
% node in a fictive pair graph. Two nodes are adjacent in the pair graph if 
% they represent contact point pairs that involve a common spartial 
% discretization point (thus linked by adhesion). Then connected components
% in the pair graph (here referred to as clusters) jointly form an adhesive
% joint in the resulting adhered fiber structure. The information is
% stored in the pair matrix which has the following column assignment:
% 1. number fiber, 
% 2. number point (global),
% 3. number contactfiber,
% 4. number contactpoint (global)
% 5. nodeNumber (pair graph) point,
% 6. nodeNumber (pair graph) contact point
% 7. cluster number pair

% allocate space
NFib = NFib_sld + NFib_adh;
pair = zeros(NFib^2,7); 


%%
% pair graph 1: (contact detection)
% compute columns 1, 2, 3, 4 of pair matrix: identify fibers and contact points 

% squared contact threshold
EPS = kappa;
EPS2 = kappa^2;
                        
% bounding boxes for detection of contacts
bBox = zeros(NFib,6);
for j=1:NFib
    bBox(j,1) = min(eta(startFib(j):endFib(j),1))-0.5*EPS;
    bBox(j,2) = max(eta(startFib(j):endFib(j),1))+0.5*EPS;
    bBox(j,3) = min(eta(startFib(j):endFib(j),2))-0.5*EPS;
    bBox(j,4) = max(eta(startFib(j):endFib(j),2))+0.5*EPS;
    bBox(j,5) = min(eta(startFib(j):endFib(j),3))-0.5*EPS;
    bBox(j,6) = max(eta(startFib(j):endFib(j),3))+0.5*EPS;
end

% detection of contacts
ptr = 1;
for j=1:NFib_adh
    for k = (j+1):NFib
        if    ((bBox(j,1) < bBox(k,1) && bBox(k,1) < bBox(j,2))...
            || (bBox(k,1) < bBox(j,1) && bBox(j,1) < bBox(k,2)))...
            && ((bBox(j,3) < bBox(k,3) && bBox(k,3) < bBox(j,4))...
            || (bBox(k,3) < bBox(j,3) && bBox(j,3) < bBox(k,4)))...
            && ((bBox(j,5) < bBox(k,5) && bBox(k,5) < bBox(j,6))...
            || (bBox(k,5) < bBox(j,5) && bBox(j,5) < bBox(k,6)))        
            d2 = EPS2;
            for p=startFib(j):endFib(j)
                for q=startFib(k):endFib(k)
                    tmp = sum((eta(p,:)-eta(q,:)).^2);
                    if(tmp < d2)
                        d2 = tmp;
                        pMin = p;
                        qMin = q;
                    end
                end
            end
            if d2 < EPS2 
                % columns 1..4 of pair
                pair(ptr,1) = j; pair(ptr,2) = pMin; 
                pair(ptr,3) = k; pair(ptr,4) = qMin;
                ptr = ptr+1;
                pair(ptr,1) = k; pair(ptr,2) = qMin; 
                pair(ptr,3) = j; pair(ptr,4) = pMin;
                ptr = ptr+1;
            end
        end   
    end
end

% number of pairs and size correction of matrix pair
NPair = ptr-1;
pair = pair(1:NPair,:);

% error handling
if size(pair,1) == 0
    disp('gra without result: no pairs found')
    return
end


%%
% pair graph 2: (cluster identification)
% compute columns 5, 6 of pair matrix: enumeration of detected contact points. 

% sortinf w.r.t. points number and therefor also w.r.t. fiber number
[~, I] = sort(pair(:,2));
pair = pair(I,:);

% for all entries in matrix pairs starting with row 2
nodeNumber = 1; pair(1,5) = nodeNumber;
for j = 2:NPair
    % if the point number changes
    if pair(j,2) ~= pair(j-1,2)
        nodeNumber = nodeNumber+1;
    end
    % set nodeNumber of point
    pair(j,5) = nodeNumber; 
end
% number of nodes of pair graph
NNodePair = nodeNumber;

% for all entries (row j) in matrix pairs
for j = 1:NPair
    % find all points with pointsnumber of contact point in row j and get
    % the corresponding nodeNumber
    tmp = (pair(pair(:,2)==pair(j,4),5));
    % set nodeNumber of contact point (all values of tmp are identical))
    pair(j,6) = unique(tmp); 
end


%%
% pair graph 3: (indentify contact point clusters)
% compute column 7 of pair matrix: introduce cluster numbers of pair graph

% adjacency matrix of undirected pair graph
APair = sparse(pair(:,5),pair(:,6),1,NNodePair,NNodePair,NPair);

% perform a depth first search to identify all connected components (cluster)
[dfs_path, ~, ~, NCluster, ComponentStartInDfstree] = DepthFirstSearch(APair,1);

% determine nodes in cluster
for i = 1:NCluster
    if i < NCluster
        NodesInCluster = unique(dfs_path(ComponentStartInDfstree(i):ComponentStartInDfstree(i+1)-1));
    elseif i == NCluster
        NodesInCluster = unique(dfs_path(ComponentStartInDfstree(i):end));
    end
    pair(ismember(pair(:,5),NodesInCluster),7) = i;
end

% output success
fprintf('Pair graph is set up. \n')

%% glue graph generation
% the glue graph represents the topology of the resulting adhered fiber
% structure. Its nodes represent fiber ends and adhesive joints (nodes of 
% the pair graph), whereas the edges represent (possible multiple) fiber 
% connections between them. Thus, nodes of the directed glue graph are the 
% identified clusters of contact points and all start and end points of
% fibers. They are numbered by 1,..,NNode. To avoid doubling one has to 
% identify start and end points that are involved in contacts. The 
% coordinates (mean values of the coordinates of the corresponding points) 
% and the type (decision by higher type) of the clusters are stored in 
% node_coordinate (size(node_coordinate)=(NNode,3)) and node_type 
% (size(node_type)=(NNode,1)).


%%
% glue graph 1: (preparation - start and end points in pair)

startInPair = zeros(NFib,1);
for j=1:NFib
    % compare point number startFib(j) with fiber number (column 2) of all
    % points in pair
    if sum(pair(pair(:,2) == startFib(j),2)) ~= 0
        startInPair(j)=1;
    end
end
endInPair = zeros(NFib,1);
for j=1:NFib
    % compare point number endFib(j) with fiber number (column 2) of all
    % points in pair
    if sum(pair(pair(:,2) == endFib(j),2)) ~= 0
        endInPair(j)=1;
    end
end


%%
% glue graph 3: (cluster coordinate and type)
% identify the cluster points of the pair graph plus start and end points 
% as nodes of glue graph. each is assigned by a coordinate and a type

% cluster points - indenfify cluster coordinates and types
node_coordinate = zeros(NCluster,3); node_type = zeros(NCluster,1);
% for each cluster j
for j=1:NCluster
    % fiber number and point number of all points in cluster j
    K = pair(pair(:,7) == j,1:2); 
    % for each point in cluster j
    for k=1:size(K,1)
        % check if point is a starting point
        if sum(K(k,2) == startFib)  
            % decision for cluster by higher type
            node_type(j) = max(node_type(j),startType(K(k,1))); 
        end
        % check if point is a end point
        if sum(K(k,2) == endFib)
            % decision for cluster by higher type
            node_type(j) = max(node_type(j),endType(K(k,1)));   
        end
        node_coordinate(j,:) = node_coordinate(j,:) + eta(K(k,2),:);
    end
    node_coordinate(j,:) = node_coordinate(j,:)/size(K,1); % mean value
end

% start and end points not in pair - coordinates and types
node_type       = [node_type;...
                   startType(~startInPair);...
                   endType(~endInPair)];
node_coordinate = [node_coordinate;...
                   eta(startFib(~startInPair),:);...
                   eta(endFib(~endInPair),:)];
NNode = size(node_type,1);


%%
% glue graph 2: (nodes on fibers)
% The matrix nodeOnFiber analyses the nodes of the glue graph w.r.t. the
% points belonging to a node under consideration. For each cluster the
% points in the matrix pair belonging to the considered cluster lead to
% entries (rows in the matrix nodeOnFiber), iff they belong to different
% fiber numbers. In other words, an entry handles all points belonging to
% the same cluster and the same fiber. Such an entry consist of 4 columns:
% the fiber number, the cluster number, the fiber number of the first, and
% the number of the last point in this list (first and last w.r.t. fiber
% number). The two points characterize the so-called glue area. A
% corresponding information is added w.r.t. the start and end points which
% are not element of any cluster. Compute nodeOnFiber matrix with columns:
% 1. fiber number
% 2. node number glue graph
% 3. point number - start of glue area on fiber (global) (:,3)
% 4. point number - end of glue area on fiber (global) (:,4)

% cluster points - entries in nodeOnFiber
tmp = [pair(:,1), pair(:,7), pair(:,2), pair(:,2)];
nodeOnFiber = zeros(size(tmp)); ptr = 0;

% for each fiber record the occuring nodes (changed from RW version)
for j=1:NFib
    I = find(tmp(:,1)==j); % rows in tmp on fiber j
    if ~isempty(I)
        ptr = ptr+1;
        nodeOnFiber(ptr,:) = tmp(I(1),:);
        for i=2:size(I,1)
            if tmp(I(i),2)~=tmp(I(i-1),2)
                ptr = ptr+1;
                nodeOnFiber(ptr,:) = tmp(I(i),:);
            else
                nodeOnFiber(ptr,4) = tmp(I(i),4);
            end
        end
    end
end
nodeOnFiber = nodeOnFiber(1:ptr,:);      

% start and end points not in pair  - entries in nodeOnFiber
tmp = (1:NFib)';
counterStartPointNodes = ((NCluster+1):...
                          (NCluster+sum(~startInPair)))';
counterEndPointNodes   = ((NCluster+sum(~startInPair)+1):...
                           NCluster+sum(~startInPair)+sum(~endInPair))';
nodeOnFiber = [nodeOnFiber;... 
               tmp(~startInPair),...
               counterStartPointNodes,...
               startFib(~startInPair), startFib(~startInPair);... 
               tmp(~endInPair),...
               counterEndPointNodes,...
               endFib(~endInPair), endFib(~endInPair)];

% sort nodeOnFiber by point number (therefore also by fiber number)
[~, I] = sort(nodeOnFiber(:,3));
nodeOnFiber = nodeOnFiber(I,:);
NNodeOnFiber = size(nodeOnFiber,1);


%%
% glue graph 4: incidence matrix

% the logical vector tmp indicates a change in the fiber number in
% nodeOnFiber
tmp = (nodeOnFiber(1:(NNodeOnFiber-1),1) == nodeOnFiber(2:NNodeOnFiber,1));
startEdge = nodeOnFiber([tmp; false],2);
endEdge = nodeOnFiber([false; tmp],2);
NEdge = length(startEdge);
B_undirected = sparse(startEdge,1:NEdge,1,NNode,NEdge,NEdge)...
    + sparse(endEdge,1:NEdge,1,NNode,NEdge,NEdge);

% remove multiple edges
B_undirected = unique(B_undirected','rows')';
[NNode,NEdge] = size(B_undirected);

% impose arbitrary edge direction
%B_directed = spalloc(NNode,NEdge,2*NEdge);
Bi = zeros(2*NEdge,1); 
Bj = zeros(2*NEdge,1); 
Bv = zeros(2*NEdge,1); 
for i = 1:NEdge
   EdgeEnds = find(B_undirected(:,i) == 1);
   Bi(2*(i-1)+1) = EdgeEnds(1); Bi(2*i) = EdgeEnds(2);
   Bj(2*(i-1)+1) = i; Bj(2*i) = i;
   Bv(2*(i-1)+1) = 1; Bv(2*i) = -1;
   %B_directed(EdgeEnds(1),i) = 1;
   %B_directed(EdgeEnds(2),i) = -1;
end
B_directed = sparse(Bi,Bj,Bv,NNode,NEdge);
incidency = B_directed;


%% 
% glue graph 5: 
% generate adjacency matrix of underlying graph

% entry generated depending on edge direction
Ai = zeros(NEdge,1); 
Aj = zeros(NEdge,1);
for j=1:NEdge
    Ai(j) = (find(B_directed(:,j)==1)); 
    Aj(j) = (find(B_directed(:,j)==-1));
end
A_directed = sparse(Ai,Aj,1,NNode,NNode,NEdge); % [Ai,Aj,~] = find(A);

% Adjacency matrix of the underlying graph
adjacency = A_directed + A_directed';


%% 
% glue graph 6: 
% glue graph coordinate transformation adjust the postion of fiber points 
% belonging to one of the clusters. Therefore, the original point position 
% is replaced by the coresponding node coordinate.

% replace adhesive node coordinate in with spartial points in eta
eta_adjusted = eta;
for k = 1:length(pair)
    eta_adjusted(pair(k,2),:) = node_coordinate(pair(k,7),:);
end

% assure sorting by fiber number and starting point of the cluster on fiber
nodeOnFiber = sortrows(nodeOnFiber, [1,3]);
prev_fib = 0;
prev_pt = 0;
for i = 1:length(nodeOnFiber)
    if nodeOnFiber(i,3) > nodeOnFiber(i,4)
        error('Empty Node Occured! \n')
    elseif nodeOnFiber(i,1) < prev_fib || nodeOnFiber(i,3) < prev_pt
        error('Sorting error! \n')
    end
    prev_fib = nodeOnFiber(i,1);
    prev_pt = nodeOnFiber(i,3);
end


%%
%%% glue graph 7: 
% Create fiber_on_edge matrix which assigns each fiber connection the edge
% it is represented by. Here an edge indicates possible multiple fiber 
% connections between nodes, for which several fibers might be assigned to
% the same edge. Additionally, we store the fiber length information for
% which we set up the fiber_on_edge matrix with the column entries:
% 1. fiber connection number
% 2. underlying edge number
% 3. length of fiber (including distance to corresponding nodes)

% Initialize fiber_on_edge matrix generation
NFiberConnections = 0;
fiber_on_edge = zeros(NNodeOnFiber,3);
PointDistances = sqrt(sum((eta_adjusted(2:end,:) - eta_adjusted(1:(end-1),:)).^2,2));

%loop through nodeOnFiber
for i = 2:NNodeOnFiber
    
    % Fiber Numbers
    previousfiber = nodeOnFiber(i-1,1);
    currentfiber = nodeOnFiber(i,1);

    % Nodes are linked by a fiber connection if they lie on the same fiber
    if previousfiber == currentfiber
        
        % update fiber connection number
        NFiberConnections = NFiberConnections + 1;
        fiber_on_edge(NFiberConnections,1) = NFiberConnections;
        
        % determine underlying edge (FiCo = fiber connection)
        FiCoStartNode = nodeOnFiber(i-1,2);
        FiCoEndNode = nodeOnFiber(i,2);
        underlyingEdge =  find(all( B_undirected([FiCoStartNode,FiCoEndNode],:) == 1));
        fiber_on_edge(NFiberConnections,2) = underlyingEdge;
        
        % determine fiber length
        FromPointOnFiber = nodeOnFiber(i-1,4);
        ToPointOnFiber = nodeOnFiber(i,3);
        CurrentFiberLength = sum(PointDistances(FromPointOnFiber:ToPointOnFiber-1));
        fiber_on_edge(NFiberConnections,3) = CurrentFiberLength;
        
        % error handeling 
        EdgeLength = sqrt(sum((eta_adjusted(nodeOnFiber(i-1,4),:) - eta_adjusted(nodeOnFiber(i,3),:)).^2,2));
        EdgeLength_alt = sqrt(sum((node_coordinate(FiCoStartNode,:) - node_coordinate(FiCoEndNode,:)).^2,2));
        if EdgeLength > CurrentFiberLength || EdgeLength_alt > CurrentFiberLength
            error('Strain Detected! \n')
        elseif FromPointOnFiber >= ToPointOnFiber
            error('Nodes Intersect on Fiber! \n')
        end
        
    end
    
end

% fiber on edge matrix truncation
fiber_on_edge = fiber_on_edge(1:NFiberConnections,:);

% output success
fprintf('Glue graph is set up. \n')

end