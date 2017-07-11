%% Manual Distance Matrix
%
% Number of vertices
nV = 24;
% Distance matrix
M = zeros(nV);
%}
%% Input distance matrix entries (Km)
%
%{
Data taken from Google Maps.
M is symmetric
%}
M(1,2) = 1.21;

M(2,3) = 1.00;
M(2,7) = 2.08 + 0.436 + 0.05;

M(3,4) = 0.262;

M(4,5) = 0.257;
M(4,10) = 1.23;

M(6,7) = 0.378;
M(6,8) = 0.464;

M(7,9) = 0.353;

M(8,9) = 0.091;
M(8,12) = 0.421;

M(9,10) = 0.006 + 1.34 + 0.01 + 0.638 + 0.304 + 0.004;

M(10,17) = 0.039 + 0.977;

M(11,13) = 0.206 + 0.267 + 0.045 + 0.257 + 0.164;

M(12,15) = 0.811;
M(12,16) = 1.1085;

M(13,14) = 0.145;
M(13,18) = 0.862;

M(14,15) = 0.248;
M(14,21) = 0.635;

M(15,19) = 0.769;

M(16,19) = 0.765;
M(16,20) = 0.739;

M(17,20) = 1.23;
M(17,23) = 0.928 + 0.528 + 0.037;

M(18,21) = 0.628;

M(19,22) = 0.143;

M(20,23) = 0.4;

M(21,22) = 0.683;

M(22,24) = 0.906;

M(23,24) = 0.181;


M = M + M';
%}

%% Convert to sparse matrix
%
M = sparse(M);
%}

%% Get Coordinates of Nodes
%
Coords = csvread('Map Manual Graph2.csv');
%}

%% Plot coordinates on a geographical map
%{
figure()
landareas = shaperead('landareas.shp','UseGeoCoords',true);
%axesm ('pcarree', 'Frame', 'on', 'Grid', 'on');
geoshow(landareas,'FaceColor',[0.5 1.0 0.5],'EdgeColor',[.6 .6 .6]);
geoshow(Coords(:,1), Coords(:,2), 'DisplayType', 'point')

xlabel('Longitude')
ylabel('Latitude')
%}

%% Create graph
%
G = graph(M);
Deg = degree(G);
Inc = incidence(G);
%}

%% Save matrices and coordinates
%
save('DistanceMatrix','M','G','Deg','Inc')
%}
%% Visualise graph
%{
% Plot as planar graph
figure()
plot(G);
title('Lappeenranta Snowplow Graph')
% Look at the structure of the adjacency matrix
%spy(Inc);

% Look at the degrees of the nodes
%bar(Deg);

%}

