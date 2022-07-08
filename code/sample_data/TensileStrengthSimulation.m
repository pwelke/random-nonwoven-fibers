function [strain,stress,stepsizes,equilibirum_deviation] = TensileStrengthSimulation(TestVolume, targetStrain, regFriction, regMaterialLaw, Output)
% TENSILESTRENGTHSIMULATION Perform tensile strength simulation on specimen

%%% DESCRIPTION:
% Uses an implicit euler scheme to perform the tensile strength simulation.
% The underlying differential equation is tailored to the considered fiber
% structure sample. For solving the nonlinear system arising in each
% implicit euler step we use a newton method with armijo step size control.

%%% INPUT: 
% TestVolume = Fiber structure sample with carried out precomputations
% targetStrain = Maximal considered strain for the simulation
% regFriction = friction-based regularization parameter
% regMaterialLaw = reg. parameter to increase material law regularity

%%% OUTPUT:
% strain = vector of strains 
% stress = vector of correpsonding stress (yields stress-strain curve)
% stepsize = stepsizes used throughout the tensile strength simulation
% equilibirum_deviation = dev from force equilibirum during the simulation


%%
% shift specification
SetUpShiftDirection(TestVolume, targetStrain);
MaterialLawRegularization(TestVolume, regMaterialLaw);
MaxStepSize = 1e-2;
MinStepSize = 1e-8;


%%
% allocate space for results
NPossibleSteps = round(1/MinStepSize);
strain = zeros(1,NPossibleSteps);
stress = zeros(1,NPossibleSteps);
stepsizes = zeros(1,NPossibleSteps);
equilibirum_deviation = zeros(1,NPossibleSteps);


%%
% ode right side declaration
RightSide = @(x,t) (1/regFriction) * NodeForceFunction(x, t, TestVolume);


%%
% integrator initiation
tk = 0;
xk = TestVolume.InitVariableNodePos;
shift_size_red_possibel = 1;
newton_iters = 4;
refinements = 0;
linelength = 0;
cnt = 0;
progress = 0;
nsteps = 0;
h = 1e-5;

%%
% display statz
fprintf('Start simulating the nonwovens mechanical behavior \n')

%%
% solve differential equation system using an implicit euler scheme
while tk < 1
    
    % stepsize controll oriented at newton method performance
    if newton_iters < 8 && h < MaxStepSize && cnt > 15
        h = 2*h;
        cnt = 0;
        if h >= 2*MinStepSize
            shift_size_red_possibel = 1;
        end
    elseif newton_iters >= 15 && h > MinStepSize
        h = 0.5*h;
        cnt = 0;
        if h < 2*MinStepSize
            shift_size_red_possibel = 0;
        end
    else
        cnt = cnt + 1;
    end
    
    % store previous iterands for reiterating
    xk_prev = xk;
    tk_prev = tk;
    progress_prev = progress;
        
    % compute new iterate
    [xk,tk,zero_deviation,newton_iters,refinement_needed] = ImplicitEulerStep(RightSide,xk_prev,tk_prev,h, regFriction, TestVolume);
    
    % decide if step is taken
    if refinement_needed && shift_size_red_possibel
        
        % if not reiterate with finer shifts
        xk = xk_prev;
        tk = tk_prev;
        progress = progress_prev;
        refinements = refinements + 1;
        if Output
            fprintf("... refining stepsize \n")
        end
        
    % error handling of step size breakdown
    elseif refinement_needed && ~shift_size_red_possibel

        % output an error
        error('Vanishing stepzite for implicite Euler method');
    
    % take the current step
    else
        
        % update iterands
        nsteps = nsteps+1;
        strain(nsteps+1) = ((TestVolume.height_init + tk * TestVolume.ShiftExtend) - TestVolume.height_init)/TestVolume.height_init;
        BoundaryNodeForce = UpperBoundaryNodeForces(xk, tk, TestVolume);
        stress(nsteps+1) = sum(BoundaryNodeForce(3:3:end));
        equilibirum_deviation(nsteps+1) = norm(NodeForceFunction(xk, tk, TestVolume));
        stepsizes(nsteps+1) = h;
        progress = tk * 100;
        
        % display progress
        if ceil(progress) > ceil(progress_prev) && Output
            fprintf('Progress: %.3f%%, Zero-Dev: %e, Equilib-Dev: %e, Newton-Steps: %i, Shift-Size: %e \n', progress, zero_deviation, equilibirum_deviation(nsteps+1), newton_iters, h);
        end
           
    end

end

% storage matrix truncation
strain = strain(1:nsteps+1);
stress = stress(1:nsteps+1);
stepsizes = stepsizes(1:nsteps+1);
equilibirum_deviation = equilibirum_deviation(1:nsteps+1);

end

% individual fiber material law
function stress = StressStrainRelation(l, L, delta)
%%
% strain computation
strain = (l - L) ./ L;

% stress computation
stress = (strain >= -delta & strain <= delta) .* (-(1/(16*delta^3)) * strain.^4 + (3/(8*delta)) * strain.^2 + (1/2)*strain + (3*delta)/16) + (strain > delta) .* strain;

end

% compute forces acting on the individual nodes
function [fx,ForceOnEdges] = NodeForceFunction(x, t, TestVolume)
    
    % compuatation of node distances and edge lenghts depending on variables x
    BoundaryNodes = TestVolume.InitBoundaryNodePos + t * TestVolume.ShiftDirection * TestVolume.ShiftExtend;
    NodeDistances = (-TestVolume.ExtIncidence)' * ( BoundaryNodes + transpose(TestVolume.ExtReductionMatrix) * x);
    SquaredDistances = NodeDistances.^2;
    EdgeLengths = (SquaredDistances(1:3:end) + SquaredDistances(2:3:end) + SquaredDistances(3:3:end)).^(0.5);
    
    % assignment of the edge lengths to the fibers representing the edges
    FiberEdgeLengths = transpose(TestVolume.FiberOnEdgeMatrix) * EdgeLengths;
    
    % computation of the stress of each fiber (vector of size n_fib x 1 )
    Stress = StressStrainRelation(FiberEdgeLengths, TestVolume.FiberLengths, TestVolume.delta);
    
    % Force on Edges (sum of stress of all fibers representing an edge)
    ForceOnEdges = TestVolume.FiberOnEdgeMatrix * Stress;
    
    % direction of the forces
    NormalizedEdgeDirection = zeros(size(NodeDistances)) ;
    NormalizedEdgeDirection(1:3:end) = (NodeDistances(1:3:end) ./ EdgeLengths) ;
    NormalizedEdgeDirection(2:3:end) = (NodeDistances(2:3:end) ./ EdgeLengths) ;
    NormalizedEdgeDirection(3:3:end) = (NodeDistances(3:3:end) ./ EdgeLengths) ;  
    NormalizedEdgeDirection(isnan(NormalizedEdgeDirection))=0;
    
    % directed forces on the edges
    DirectedForces = zeros(size(NodeDistances));
    DirectedForces(1:3:end) = NormalizedEdgeDirection(1:3:end) .* ForceOnEdges;
    DirectedForces(2:3:end) = NormalizedEdgeDirection(2:3:end) .* ForceOnEdges;
    DirectedForces(3:3:end) = NormalizedEdgeDirection(3:3:end) .* ForceOnEdges;
    
    % resulting force on the inner nodes
    fx = TestVolume.ExtVariableNodeFiberDirections * DirectedForces; 
    
end

% compute stress acting on upper boundary nodes
function UpperBoundaryNodeForce = UpperBoundaryNodeForces(x, t, TestVolume)

    % compuatation of node distances and edge lenghts depending on variables x
    BoundaryNodes = TestVolume.InitBoundaryNodePos + t * TestVolume.ShiftDirection * TestVolume.ShiftExtend;
    NodeDistances = (-TestVolume.ExtIncidence)' * ( BoundaryNodes + transpose(TestVolume.ExtReductionMatrix) * x);
    SquaredDistances = NodeDistances.^2 ;
    EdgeLengths = (SquaredDistances(1:3:end) + SquaredDistances(2:3:end) + SquaredDistances(3:3:end)).^(0.5) ;
    
    % assignment of the edge lengths to the fibers representing the edges
    FiberEdgeLengths = transpose(TestVolume.FiberOnEdgeMatrix) * EdgeLengths;
    
    % computation of the stress of each fiber (vector of size n_fib x 1 )
    Stress = StressStrainRelation(FiberEdgeLengths, TestVolume.FiberLengths, TestVolume.delta);
    
    % force on edges (sum of stress of all fibers representing an edge)
    ForceOnEdges = TestVolume.FiberOnEdgeMatrix * Stress;

    % direction of the forces
    NormalizedEdgeDirection = zeros(size(NodeDistances)) ;
    NormalizedEdgeDirection(1:3:end) = (NodeDistances(1:3:end) ./ EdgeLengths) ;
    NormalizedEdgeDirection(2:3:end) = (NodeDistances(2:3:end) ./ EdgeLengths) ;
    NormalizedEdgeDirection(3:3:end) = (NodeDistances(3:3:end) ./ EdgeLengths) ;    
    NormalizedEdgeDirection(isnan(NormalizedEdgeDirection))=0;
    
    % directed forces on the edges
    DirectedForces = zeros(size(NodeDistances));
    DirectedForces(1:3:end) = NormalizedEdgeDirection(1:3:end) .* ForceOnEdges;
    DirectedForces(2:3:end) = NormalizedEdgeDirection(2:3:end) .* ForceOnEdges;
    DirectedForces(3:3:end) = NormalizedEdgeDirection(3:3:end) .* ForceOnEdges;

    % resulting force on the inner nodes
    UpperBoundaryNodeForce = TestVolume.ExtUpperBoundaryNodeFiberDirections * DirectedForces; 

end

% compute jacobian of forces acting on the individual nodes
function DFdx = ActiveNodeForceJacobian(x, t, TestVolume)
% Sparse computation of the jacobian by applying chain rule on the node force
% function. We write e,l,d,... for functions and d_x,l_dx,.. for their 
% evaluation, i.e., l_dx = l(d(x)). According to this we write De_ldx for 
% the jacobian of e evaluated at l(d(x)). 

% get structural information
L = TestVolume.FiberLengths;
NFib = length(L);
[~, NEdg] = size(TestVolume.Incidence);


%%
% Computing the point distances and the edge lengths

% edge distances d(x) = (BxI_3 ) x and derivative
BoundaryNodes = TestVolume.InitBoundaryNodePos + t * TestVolume.ShiftDirection * TestVolume.ShiftExtend;
dx = (-TestVolume.ExtIncidence)' * (BoundaryNodes + TestVolume.ExtReductionMatrix' * x);
Dd_x = -TestVolume.mp1;

% edge lengths
dx2 = dx.^2 ;
l_dx = (dx2(1:3:end) + dx2(2:3:end) + dx2(3:3:end)).^(0.5);


%%
% Computing edgeStrain and derivative DedgeStrain

% lengths of edge belonging to fiber i = 1,..,NFib
lf_dx = TestVolume.FiberOnEdgeMatrix' * l_dx;

% determine relevant edges
IsStreched = (lf_dx > (1-TestVolume.delta)*L);
ActiveEdges = find(TestVolume.FiberOnEdgeMatrix * IsStreched);
NActiveEdges = length(ActiveEdges);
I = ones(NActiveEdges*3,1);
J = ones(NActiveEdges*3,1);
V = zeros(NActiveEdges*3,1);

% iteration through columns corresponding to active edges
for k = 1:NActiveEdges 
    
    % check for edge uder stress
    i = ActiveEdges(k);
    
    % row indices
    I((k-1)*3+1) = i;
    I((k-1)*3+2) = i;
    I(k*3) = i;
    
    % column indices
    J((k-1)*3+1) = (i-1)*3+1;
    J((k-1)*3+2) = (i-1)*3+2;
    J(k*3) = i*3;
    
    % insert entries of derivative matrices
    l_i = sqrt(dx2(3*(i-1)+1) +dx2(3*(i-1)+2) +dx2(3*(i-1)+3));
    V(3*(k-1)+1) = dx(3*(i-1)+1)/l_i;
    V(3*(k-1)+2) = dx(3*(i-1)+2)/l_i;
    V(3*(k-1)+3) = dx(3*(i-1)+3)/l_i;
    
end
Dl_dx = sparse(I,J,V,NEdg,3*NEdg);
Dlf_dx = TestVolume.FiberOnEdgeMatrix' * Dl_dx;

% fiber strains and derivatives
diagLinv = TestVolume.spdiag1;
e_ldx = diagLinv*(lf_dx - L); % strain = e(l(d(x)))
De_ldx = diagLinv;

% Jacobian of the fiberstress depending on strain eps
V = zeros(NFib,1);
delta = TestVolume.delta;

% Elementwise force computation
StrechedFibers = find(IsStreched);
NStrechedFibers = length(StrechedFibers);
for k = 1:NStrechedFibers
    i = StrechedFibers(k);
    if e_ldx(i) > delta
        V(i) = 1;
    elseif e_ldx(i) >= - delta && e_ldx(i) <= delta
        V(i) = (-1/(4*delta^3)) * e_ldx(i)^3 + (3/(4*delta)) * e_ldx(i) + 0.5;
    end
end
DN_eldx = spdiags(V,0,NFib,NFib);
DN_eldx = kron(TestVolume.FiberOnEdgeMatrix * DN_eldx,ones(3,1));

% compute jacobian of the edge strain via chain rule
edgStrain = kron(TestVolume.FiberOnEdgeMatrix * StressStrainRelation(lf_dx, L, delta),ones(3,1));
DedgStrain = ((DN_eldx * De_ldx) * Dlf_dx) * Dd_x;


%%
% Compute force direction and derivative Da_dx

% acting force direction = a (d(x))
I = ones(9*NActiveEdges,1);
J = ones(9*NActiveEdges,1);
V = zeros(9*NActiveEdges,1);

% iteration through columns corresponding to active edges
for k = 1:NActiveEdges  
    
    % check for edge uder stress
    i = ActiveEdges(k);
    
    % iterating through 3x3 block matrices
    b = 3*(i-1);
    l_b = (dx2(b+1) +dx2(b+2) +dx2(b+3) )^(1.5);
    
    % row indices
    I((k-1)*9+1)= b+1;
    I((k-1)*9+2)= b+1;
    I((k-1)*9+3)= b+1;
    I((k-1)*9+4)= b+2;
    I((k-1)*9+5)= b+2;
    I((k-1)*9+6)= b+2;
    I((k-1)*9+7)= b+3;
    I((k-1)*9+8)= b+3;
    I(k*9)= b+3;
    
    % column indices
    J((k-1)*9+1) = b+1;
    J((k-1)*9+4) = b+1;
    J((k-1)*9+7) = b+1;
    J((k-1)*9+2) = b+2;
    J((k-1)*9+5) = b+2;
    J((k-1)*9+8) = b+2;
    J((k-1)*9+3) = b+3;
    J((k-1)*9+6) = b+3;
    J(k*9) = b+3;
    
    % generate derivative entries.
    V((k-1)*9+1) = (dx2(b+2) + dx2(b+3)) / l_b;
    V((k-1)*9+2) = -(dx(b+1)*dx(b+2))/ l_b;
    V((k-1)*9+3) = -(dx(b+1)*dx(b+3))/ l_b;
    V((k-1)*9+4) = -(dx(b+2)*dx(b+1))/ l_b;
    V((k-1)*9+5) = (dx2(b+1) + dx2(b+3)) / l_b;
    V((k-1)*9+6) = -(dx(b+2)*dx(b+3))/ l_b;
    V((k-1)*9+7) = -(dx(b+3)*dx(b+1))/ l_b;
    V((k-1)*9+8) = -(dx(b+3)*dx(b+2))/ l_b;
    V(k*9) = (dx2(b+1) + dx2(b+2)) / l_b;
    
end
Da_dx = sparse(I,J,V,3*NEdg,3*NEdg);

% compute jacobian of the force direction via chain rule
Ddir = Da_dx * Dd_x;
dir = kron(spdiags(l_dx.^(-1),0,NEdg,NEdg),eye(3)) * dx;


%%
% Computing derivative of inner node force

% elementwise product rule on sparse triplet level
[srow,sclm]= size(DedgStrain);
[i,j,a]=find(DedgStrain);
b=dir(i);
elsum1=sparse(i,j,a.*b,srow,sclm);
[i,j,a]=find(Ddir);
b = edgStrain(i);
elsum2=sparse(i,j,a.*b,srow,sclm);
Dndx = elsum1+elsum2;

% final assembly of node force jacobian
DFdx = TestVolume.mp2 * Dndx;

end

% implicit euler step for tensile strength simulation
function [x_new,t_new,dev,newton_iters,refine] = ImplicitEulerStep(func,x_prev,t_prev,h, beta, TestVolume)

% explicit euler prediction
x_pred = x_prev + h *func(x_prev,t_prev);

% set up nonlinear equation system and perform implicit euler correction
t_new = t_prev + h;
G = @(x) x - x_prev - h * func(x,t_new); 
[x_new,dev,newton_iters,refine]= Newton4ode(G, x_pred, t_new, TestVolume, h, beta);

end

% newton-simpson method to solve nonliner implicit euler systems
function [xk,norm_fk,k,refine] = Newton4ode(func, x_init, t_new, TestVolume, h, beta)

% armijo-rule rarameter
rho = 0.5;
gamma = 1e-4;
tau_init = 1;

% initialization
xk = x_init;
fk = func(x_init);
norm_fk = norm(fk);
k = 0;
refine = 0;
iteration_limit = 50;

% exact newton method
while k <= iteration_limit && norm_fk >= 1e-8

    % compute jacobian and enforcy symmetry
    Jk = speye(length(xk)) - (h/beta) * ActiveNodeForceJacobian(xk, t_new, TestVolume);
    Jsym = triu(Jk) + triu(Jk,1)';
    
    % solve newton system for newton step
    dx = Jsym\-fk;
    if sum(isnan(dx)) > 0
        error('A NaN Error Occured')
    end
    
    % armijo backtracking
    tau = tau_init;
    while norm(func(xk + tau*dx)) > (1 - gamma * tau) * norm_fk
        tau = rho*tau;
    end
    
    % update iterands
    xk = xk + tau *  dx;
    fk = func(xk);
    norm_fk  = norm(fk);
    k = k+1;
    
    % trigger refinement for implicit euler step size (redo with smaller h)
    if k == iteration_limit
        refine = 1;
    end
    
end

end
