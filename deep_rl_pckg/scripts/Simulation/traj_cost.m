function f = traj_cost(z, opt)

xstemp = traj(z, opt);

%add cost function here

%--------------------TODO-----------------%
%%
if (opt.planner==1)
%1)Utility maximization
%find utility:opt.AF along the trajectory
 [traj_utility] = EvaluateTrajUtility(xstemp,opt);
 f = -sum(traj_utility);
% % % get the info of the closest points to the trajectory in a precomputed info map:computationally expensive
% % kdOBJ = KDTreeSearcher(opt.info(:,1:opt.dim));
% % [idxInfo, ~] = knnsearch(kdOBJ,xstemp(1:opt.dim,:)');
% % f =-sum(opt.info(idxInfo,opt.dim+1));
end

%%
if (opt.planner==2)
%%2)Ergodic coverage
f = EvaluateErgodicity(xstemp(1:opt.dim,:),opt);
end
 %--------------------------------%


% check for bounds
for i=1:2
    if sum(find(xstemp(i,:) < opt.xlb(i))) || sum(find(xstemp(i,:) > opt.xub(i)))
        f = 100000;
        return
    end
end

end

