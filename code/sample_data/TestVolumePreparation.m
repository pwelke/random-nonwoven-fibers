classdef TestVolumePreparation < handle
% TESTVOLUMEPREPARATION Performs necessary precompuations

%%% DESCRIPTION:
% This class contains all relevant information for the subsequent
% tensile strength simulations. This Also includes the storage of 
% scalars, vectors or matrices which are repedeatly used in order to 
% avoid the associated computational overhead
    
%%% INPUT: 
% adjacency = adjaceny of graph representing adhered fiber web topology
% incidency = incidency of graph representing adhered fiber web topology
% node_type = indicates if node is inner node (0) or lateral face node (1-6)
% node_coordinate = spartial coordinate of nodes
% fiber_on_edge = assigns fibers and their lengths to the edges

%%% OUTPUT:
% performs precomputations


%%
% class properties
    properties
        
        % graph information
        Adjacency
        Incidence
        Incidence_abs
        ExtIncidence   % kron(Incidenc,eye(3))
        n_edg
        n_nod
        

        % node information
        InitNodePos
        InitBoundaryNodePos
        InitVariableNodePos
        IsVariableNode
        IsRight
        IsLeft
        IsFront
        IsBack
        BoundaryNodeType
        dim
        
        % fiber information
        FiberLengths
        FiberOnEdgeMatrix
        n_fib
        
        % further information
        ReductionMatrix
        ExtReductionMatrix
        ExtVariableNodeFiberDirections
        ExtUpperBoundaryNodeFiberDirections
        BoundaryReductionMatrix
        
        % precomputations for the Jacobian
        mp1
        mp2
        mp3
        spdiag1
        sparse_pattern
        
        % shift specifications
        height_init
        ShiftDirection
        ShiftExtend
        
        % material law regularization
        delta
        
    end   
        
    methods
        %%
        % Constructor: performs nessecary precompuations
        function obj = TestVolumePreparation(Adjacency, Incidence, NodeTypes, NodePositions, FiberOnEdge)
            
            % spatial dimension
            obj.dim = 3;
            
            % copy graph information
            obj.Adjacency = sparse(Adjacency);
            obj.Incidence = sparse(Incidence);
            obj.Incidence_abs = sparse(abs(Incidence));
            [obj.n_nod, obj.n_edg] = size(Incidence);
            
            % get initial node positions as vector (convert from n_nodesx3 matrix)
            InitNodePositions = zeros(obj.n_nod * obj.dim,1);
            InitNodePositions(1:3:end) = NodePositions(:,1);
            InitNodePositions(2:3:end) = NodePositions(:,2);
            InitNodePositions(3:3:end) = NodePositions(:,3);
            obj.InitNodePos = InitNodePositions;
            
            % processing node type (ranges from 0 to 6)
            % free moving nodes includes lateral boundary node with type <5
            obj.IsVariableNode = (NodeTypes <= 4);
            obj.BoundaryNodeType = (NodeTypes == 6) - (NodeTypes == 5);
            VariableNodeTypes = NodeTypes(obj.IsVariableNode);
            obj.IsRight = (VariableNodeTypes == 2);
            obj.IsLeft = (VariableNodeTypes == 1);
            obj.IsFront = (VariableNodeTypes == 4);
            obj.IsBack = (VariableNodeTypes == 3);
            
            % fiber information
            obj.FiberLengths = FiberOnEdge(:,3);
            [obj.n_fib, ~] = size(FiberOnEdge);
            obj.FiberOnEdgeMatrix = spalloc(obj.n_edg, obj.n_fib, obj.n_fib);
            for i = 1:obj.n_fib
                obj.FiberOnEdgeMatrix(FiberOnEdge(i,2), FiberOnEdge(i,1)) = 1;
            end
         
            % redmat (= projection matrix) allows projetion to free moving nodes
            RedMat = spdiags(obj.IsVariableNode,0,length(obj.IsVariableNode),length(obj.IsVariableNode));  
            RedMat(sum(RedMat) == 0, :) = [];
            brm = speye(length(obj.IsVariableNode));
            brm(any(RedMat ~= 0), :) = [];
            obj.ReductionMatrix = RedMat;
            obj.BoundaryReductionMatrix = brm;
            
            % store blown up matrices to avoid multiple computation
            obj.ExtIncidence = kron(obj.Incidence, eye(obj.dim));
            obj.ExtReductionMatrix = kron(obj.ReductionMatrix, eye(obj.dim));
            
            % reducing the fiber direction matrix to directions at free moving nodes
            FiberDirectionsFromVariableNodes = obj.Incidence;
            FiberDirectionsFromVariableNodes(~obj.IsVariableNode,:) = [];
            obj.ExtVariableNodeFiberDirections = kron(FiberDirectionsFromVariableNodes, eye(obj.dim));
            
            % reducing the fiber direction matrix to directions at boundary nodes
            FiberDirectionsFromUpperBoundaryNodes = -1 * obj.Incidence;
            IsLowerBoundaryNode = (NodeTypes == 5);
            DeletableRows = logical(obj.IsVariableNode + IsLowerBoundaryNode);
            FiberDirectionsFromUpperBoundaryNodes(DeletableRows,:) = [];
            obj.ExtUpperBoundaryNodeFiberDirections = kron(FiberDirectionsFromUpperBoundaryNodes, eye(obj.dim));

            % get initial free moving node postitions
            obj.InitVariableNodePos = kron(RedMat,eye(obj.dim)) * obj.InitNodePos;
            
            % get initial boundary node posistions
            sp_aux = sparse(1:obj.n_nod,1:obj.n_nod,ones(length(obj.IsVariableNode),1) - obj.IsVariableNode,obj.n_nod,obj.n_nod);
            obj.InitBoundaryNodePos = kron(sp_aux, speye(obj.dim)) * obj.InitNodePos;
            
            % jacobian precomputations 
            obj.mp1 = obj.ExtIncidence' * obj.ExtReductionMatrix';
            obj.mp2 = obj.ExtReductionMatrix * obj.ExtIncidence;
            obj.mp3 = kron(obj.FiberOnEdgeMatrix,eye(3));
            obj.spdiag1 = spdiags(obj.FiberLengths.^(-1),0,length(obj.FiberLengths),length(obj.FiberLengths));
            
            % determine sparse pattern
            AExt = obj.ExtReductionMatrix * kron(obj.Adjacency,ones(3)) * obj.ExtReductionMatrix'; 
            obj.sparse_pattern = AExt + speye(length(AExt));
            
            % output end of precomputations
            fprintf('Cleared precompuations for tensile strength simulations \n')
            
        end
        
        %%
        % Sets up vertical shift
        function SetUpShiftDirection(obj,targetStrain)
            
            % set up vertical shift
            obj.ShiftDirection = zeros(size(obj.InitBoundaryNodePos)); 
            obj.ShiftDirection(3:3:end) = max(obj.BoundaryNodeType,0); 
            H_min = min(obj.InitNodePos(3:3:end));
            H_max = max(obj.InitNodePos(3:3:end));
            
            % initial volume heigth
            obj.height_init = H_max-H_min;
            
            % determine maximal shift extend 
            obj.ShiftExtend = obj.height_init * targetStrain;
            
        end
        
        
        %%
        % material law regularization parameter choice
        function MaterialLawRegularization(obj,regMaterialLaw)
            
            % material law regularization parameter
            obj.delta = regMaterialLaw;
            
        end
        
    end
    
end