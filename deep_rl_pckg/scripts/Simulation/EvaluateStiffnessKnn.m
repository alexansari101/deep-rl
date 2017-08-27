function [kGP] = EvaluateStiffnessKnn(posGP,gt,c)
[IDX,D] = knnsearch(gt(:,1:2),posGP(:,1:2));
kGP=c(IDX);
end

