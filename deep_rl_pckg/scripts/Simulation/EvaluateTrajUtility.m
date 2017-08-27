function [traj_utility] = EvaluateTrajUtility(xs_traj,opt)

[ymu, ys2, mu, fs2]= gp(opt.gp_para,opt.gp_model.inf, opt.gp_model.mean, opt.gp_model.cov, opt.gp_model.lik, opt.gp.posGP, opt.gp.kGP, xs_traj(1:opt.dim,:)');

if(opt.method==1)%Expected Improvement
    yEI=max(opt.gp.kGP);%current max
    traj_utility = EI(yEI,ymu,ys2);
end

if(opt.method==2)%Variance Reduction
    traj_utility = ys2;
end
if opt.method==3
    std = sqrt(fs2);
    
    Q = [ ...
      mu - sqrt(opt.method_obj.beta_t) * std, ...
      mu + sqrt(opt.method_obj.beta_t) * std       ];
    
    traj_utility = min(Q(:,2) - opt.method_obj.level, opt.method_obj.level - Q(:,1));
end

if opt.method==4
    ind = knnsearch(opt.kdOBJ, xs_traj(1:2,:)');
    traj_utility = opt.gp.AF(ind);
%     traj_utility = opt.method_obj.utility(xs_traj(1:(opt.dim-1),:)');
end

