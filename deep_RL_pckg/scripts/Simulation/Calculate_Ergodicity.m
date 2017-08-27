function [Ergodicity_Metric] = Calculate_Ergodicity(Ck, muk, DomainBounds)
Lx = DomainBounds.xmax - DomainBounds.xmin;
Ly = DomainBounds.ymax - DomainBounds.ymin;
Nkx = size(muk, 1);
Nky = size(muk, 2);
Nagents = 1;


KX = (0:Nkx-1)' * ones(1,Nky);
KY = ones(Nkx,1) * (0:Nky-1);
LK = 1.0 ./ ((1.0 + KX.^2 + KY.^2).^1.5);

Ergodicity_Metric=0;
for iagent = 1:Nagents
    Ergodicity_Metric = Ergodicity_Metric + sum(sum( LK .* (abs(Ck - muk)).^2 ));
end

end


