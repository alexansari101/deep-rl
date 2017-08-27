function [opt,xs_traj] = gen_traj_CE(opt)
  opt.ce.C = .5*opt.ce.C + .5*opt.ce.C0;
    z = opt.ce.z0;
    for i=1:opt.iters
        opt.z = z;
        [z, c, mu, C] = cem(@traj_cost, z, opt.ce, opt);
        opt.ceFlag=0; %plot trajectories inside cem
        opt.ce.C = C;
        xs_traj = traj(z, opt);    
    %plot traj in cem
% if opt.ceFlag==1;
%     set(opt.ceFig_candidate,'XData', xs_traj(1,:));
%     set(opt.ceFig_candidate,'YData', xs_traj(2,:));
%     set(opt.ceFig_candidate,'ZData', xs_traj(3,:));
%     drawnow
% end

    end
    
end