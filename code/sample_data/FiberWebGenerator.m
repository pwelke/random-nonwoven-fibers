function [eta,startFib,endFib,startType,endType,NFib_sld,NFib_adh] = FiberWebGenerator(alpha_solid, sigma_ramp, sigma_sde)
% FIBERWEBGENERATOR Samples web consisting of solid and bi-component fibers

%%% DESCRIPTION:
% seperatly samples a web of solid fibers and a web of bi-component fibers
% that are joint in the test volume of interest, which yields:

%%% INPUT: 
% alpha = number of fibers per meter in cross direction per second
% sigma_ramp = standard deviation of the distribution describing the ramp
% sigma_sde = standard deviation of fiber lay-down

%%% OUTPUT:
% eta = three-dimensional spartial discretization points of all fibers
% startFib = indicate start points of the individual fibers in eta
% endFib = indicate end points of the individual fibers in eta
% startType = indicates where fiber starts (in test volume, or outer face) 
% endType = indicates where fiber end (in test volume, or outer face)
% NFib_sld = number of sampled solid fibers
% NFib_sld = number of sampled bi-component fibers

%% Generate Joint Fiber Structure

% generate solid fiber web
[eta_sld,startFib_sld,endFib_sld,startType_sld,endType_sld, ~] = IndividualFiberWebGenerator(alpha_solid, sigma_ramp, sigma_sde);
fprintf('Solid fiber web generated. \n')

% generate bi-component fiber web
alpha_ratio = ((0.3*4.8e-2)/(4.4e-7*2.4*5.5e-2)) / ((0.7*4.8e-2)/(6.7e-7*2.4*5.5e-2));
alpha_adhesive = alpha_solid * alpha_ratio;
[eta_adh,startFib_adh,endFib_adh,startType_adh,endType_adh, ~] = IndividualFiberWebGenerator(alpha_adhesive, sigma_ramp, sigma_sde);
fprintf('Bi-component fiber web generated. \n')

% fiber numbers
NFib_sld = length(startFib_sld);
NFib_adh = length(startFib_adh);

% join fiber information:
eta = [eta_adh; eta_sld];
startFib = [startFib_adh; startFib_sld + length(eta_adh)];
startType = [startType_adh; startType_sld];
endFib = [endFib_adh; endFib_sld + length(eta_adh)];
endType = [endType_adh; endType_sld];

% plot joint fiber structure
Plot_FiberStructure = 0;
if Plot_FiberStructure
    % figure options
    figure(); 
    hold on; 
    view(120,8);
    axis equal;
 
    % plot solid fibers    
    for j = 1:NFib_sld
        plot3(eta_sld(startFib_sld(j):endFib_sld(j),1),...
               eta_sld(startFib_sld(j):endFib_sld(j),2),...
               eta_sld(startFib_sld(j):endFib_sld(j),3),'r-')
    end
    
    % plot bicomponent fibers
    for j = 1:NFib_adh
        plot3(eta_adh(startFib_adh(j):endFib_adh(j),1),...
               eta_adh(startFib_adh(j):endFib_adh(j),2),...
               eta_adh(startFib_adh(j):endFib_adh(j),3),'b-')
    end
    hold off;
end   

end

% samples individual fiber web (of solid and bi-component fibers respectivly)
function [eta,startFib,endFib,startType,endType, ds] = IndividualFiberWebGenerator(alpha, sigma_ramp, sigma_sde)
% FiberStructureGenerator: Generates random virtual fiber web using a 
% surrogare model that is based on drawing fiber ends and realizing the 
% corresponding lay-down over a stochastic process descibed by an SDE.

%%
% non-dimensional fixed parameters
H = 6e-2/1e-2;          % volume height
w = 1;                  % volume width
L = 5.5e-2/1e-2;        % fiber length
vB = 1;                 % belt movement
NP = 200;               % discretization points per fiber
B = 0.3;                % anisotropical lay-down behavior

%%
% determine joint lay-down density and the resulting ramp structure
[F, R, Rs, dx, xMin, xMax] = RampContour(H,sigma_ramp);

%%
% other dependent parameter
zMin = 0.01 * H;                                                       % fiberrange glued to the lower metal plate
zMax = 0.99 * H;                                                       % fiberrange glued to the upper metal plate
sigma1 = sigma_sde * (1.5e-2/2e-2);                                         % fiber deposit std in x-direction
sigma2 = sigma_sde * 1;                                                     % fiber deposit std in y-direction
sigma3 = sigma_sde * (1.5e-3/2e-2);                                         % fiber deposit std in z-direction
A = 1/sqrt(sigma_sde) * 0.2 * sqrt(2e-2);                                   % diffusion in lay-down behavior
wR = 1+2*L;                                                                 % width reference volume
TR = (xMax-xMin+wR);                                                        % production time   
NCon = round(alpha*wR*TR);                                                  % Total number of considered fibers
ds = L/(NP-1);                                                              % discretization length fibers

%% 
% reference points
% We consider a random variable (X,Y,t) as reference point. 
% Thereby, f is the distribution of X.
[X, Y, t] = ReferencePoints(F,xMin,xMax,vB,wR,TR,NCon,dx);

% number of fibers in reference volume
NRef = length(t); % number of fibers in reference volume

% X-Offset and Z-coordinate
xB = xMin - 0.5*wR + vB*t;
Z = R(X);


%%
% process computation in reference volume
eta_ = zeros(NRef*NP,3);
for j=1:NRef
    eta_((j-1)*NP+1:j*NP,:) = RealizeFibers(Rs, sigma1, sigma2, sigma3, A, B, NP, ds, [X(j), Y(j) Z(j)]);
    eta_((j-1)*NP+1:j*NP,1) = eta_((j-1)*NP+1:j*NP,1) - xB(j);
end


%%
% reduction to test volume
test = eta_(:,1) > -0.5*w & eta_(:,1) < 0.5*w & eta_(:,2)>-0.5*w & eta_(:,2) < 0.5*w & eta_(:,3) > zMin & eta_(:,3) < zMax;

% decide on fiber start/end type
k = 0;
startFib_ = zeros(NRef*NP,1); lengthFib = zeros(NRef*NP,1); 
startType = zeros(NRef*NP,1); endType = zeros(NRef*NP,1);
for j=1:NRef
    searchForStart = 1;
    for i = (j-1)*NP+1:j*NP
        if searchForStart && test(i)
            k = k+1;
            startFib_(k) = i;
            lengthFib(k) = j*NP-i+1;
            searchForStart = 0;
            % decision for point by higher type
            if i > (j-1)*NP+1
                if eta_(i-1,3) > zMax
                    startType(k) = 6;  
                elseif eta_(i-1,3) < zMin
                    startType(k) = 5;  
                elseif eta_(i-1,2) > 0.5*w
                    startType(k) = 4;
                elseif eta_(i-1,2) < -0.5*w
                    startType(k) = 3;
                elseif eta_(i-1,1) > 0.5*w
                    startType(k) = 2;
                elseif eta_(i-1,1) < -0.5*w
                    startType(k) = 1;       
                end
            end
            
        end
        % decision for point by higher type
        if ~searchForStart && ~test(i)
            lengthFib(k) = i-startFib_(k);
            if eta_(i,3) > zMax
                    endType(k) = 6;
            elseif eta_(i,3) < zMin
                    endType(k) = 5;
            elseif eta_(i,2) > 0.5*w
                    endType(k) = 4; 
            elseif eta_(i,2) < -0.5*w
                    endType(k) = 3;
            elseif eta_(i,1) > 0.5*w
                    endType(k) = 2;
            elseif eta_(i,1) < -0.5*w
                    endType(k) = 1;   
            end
            searchForStart = 1;
        end
    end
end
tmp = (lengthFib>1);
startFib_ = startFib_(tmp); lengthFib = lengthFib(tmp);
startType = startType(tmp); endType = endType(tmp);

% matrix truncation
NFib = length(lengthFib);
tmp = [1; 1+cumsum(lengthFib)];
startFib = tmp(1:NFib);
endFib = tmp(2:(NFib+1))-1;

% all fiber in terms of their spartial discretization points in test volume
eta = zeros(endFib(NFib),3);
for j = 1:NFib
    eta(startFib(j):endFib(j),:) = eta_(startFib_(j):startFib_(j)+lengthFib(j)-1,:);
end

end

% determines joint lay-down density and resutling ramp
function [F, R, Rs, dx, xMin, xMax] = RampContour(H,sigma)
% determine support interval for joint lay-down density
xMin = -5*sigma;    % upper bound of support interval of joint lay-down density                   
xMax = 5*sigma;     % upper bound of support interval of joint lay-down density
dx = (xMax-xMin) / 1e4;

% density
x = (xMin:dx:xMax)';
f_ = exp(-(1.0/2.0)*(x/sigma).^2);
f_(1)=0; f_(end) = 0;
C = dx*sum(f_);                 % integration with trapezoidal rule (f(0)=f(end)=0)
f_ = f_/C;                      % scaling
mu = dx*sum(x.*f_);             % mean value with trapezoidal rule
x = x-mu;                       % shifting
xMin = x(1); xMax = x(end);

% cummulative distribution given by F(x) = int_xMin^x f(x') dx'.
F_  = dx*(cumsum(f_) - 0.5*f_); % integration with trapezoidal rule (f(0)=f(end)=0)
F_(end) = 1;                    % eps-correction

% interpolants
F  = griddedInterpolant(x,F_,'linear','nearest');
R  = griddedInterpolant(x,H*F_,'linear','nearest');
Rs = griddedInterpolant(x,H*f_,'linear','nearest');
end

% samples fiber end points
function [X, Y, t] = ReferencePoints(F,xmin,xmax,vB,dR,TR,NF,dx)

% reference points are F-distributed in x and, uniform distributed in y, 

t_ = linspace(0,TR,NF)';
xB = xmin - 0.5*dR + vB*t_;

randNum = rand(NF,2);
x = xmin:dx:xmax;
X_ = interp1(F(x),x,randNum(:,1));
Y_ = dR*(randNum(:,2)-0.5);

tmp = X_-xB; inReferenceVolume = tmp>-0.5*dR & tmp<0.5*dR;
t = t_(inReferenceVolume);
X = X_(inReferenceVolume);
Y = Y_(inReferenceVolume);

end

% Samples Fiber lay-down via Euler-Maryama scheme
function [xi] = RealizeFibers(Rs, sigma1, sigma2, sigma3, A, B, N, ds, xistart)

% allocation
zeta  = zeros(N,3); % process
xi    = zeros(N,3);
alpha = zeros(N,1); % angle
theta = zeros(N,1);

% random numbers
randnum =randn(N-1,2);

% initial values
zeta(1,:)  = [0, 0, 0];
xi(1,:)  = xistart;
alpha(1) = 2*pi*rand;
theta(1) = pi/2;

% fiber points
sinalpha = sin(alpha(1)); cosalpha = cos(alpha(1)); sintheta = sin(theta(1)); costheta = cos(theta(1));
for i=1:N-1
    
    gradV = [zeta(i,1)/(sigma1*sigma1), zeta(i,2)/(sigma2*sigma2), zeta(i,3)/(sigma3*sigma3)];
    
    p = gradV*[-sinalpha; cosalpha; 0]/((B+1.0)*sintheta);
    q = gradV*[cosalpha*costheta; sinalpha*costheta; -sintheta]*B/(B+1.0);
    
    alpha(i+1) = alpha(i) - p*ds + A/sin(theta(i))*sqrt(ds)*randnum(i,1);
    theta(i+1) = theta(i) - q*ds + 0.5*A*A*cot(theta(i))*ds + A*sqrt(B)*sqrt(ds)*randnum(i,2);
    
    sinalpha = sin(alpha(i+1)); cosalpha = cos(alpha(i+1)); sintheta = sin(theta(i+1)); costheta = cos(theta(i+1)); 
    zeta(i+1,:)  = zeta(i,:) + [cosalpha*sintheta, sinalpha*sintheta, costheta]*ds;
    
    Rsloc = Rs(xi(i,1));
    xi(i+1,:) = xi(i,:) + [(cosalpha*sintheta-Rsloc*costheta)/sqrt(1.0+Rsloc*Rsloc),sinalpha*sintheta,(costheta + Rsloc*cosalpha*sintheta)/sqrt(1.0+Rsloc*Rsloc)]*ds;
     
end

end
