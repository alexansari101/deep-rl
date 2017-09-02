function [X,Y,G] = GenerateStiffnessMap_prior(xRange,yRange,addnoise)
[X,Y] = meshgrid(xRange,yRange);
if (addnoise==1)
    n=0.05;
else
    n=0;
end

m=[80 40];
s=400*eye(2);
m2=[80 120];
s2=400*eye(2);
G1 = mvnpdf([X(:), Y(:)],m,s);
G2 = mvnpdf([X(:), Y(:)],m2,s2);
G=(G1+G2);
% noise=n*rand(size(G));
% G=G+noise;
G=max(G,0); %crop below 0
G=G./max(G); %normalize
end

