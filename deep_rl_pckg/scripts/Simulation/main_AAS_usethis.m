%% Initialization
clc;close all;clear;
if(~robotics.ros.internal.Global.isNodeActive)
    rosinit('192.168.1.100')    
    setenv('ROS_MASTER_URI','http://192.168.1.100:11311')
    setenv('ROS_IP','192.168.1.101') %% this is the IP address of the machine running MATLAB
    
    getenv('ROS_MASTER_URI')
end

%%%%%%%%%%%%%%%%Definition of Variables%%%%%%%%%%%%%%%%%%%%
%xss (m x d):3-d organ grid
%sGT (m x 1):stiffness ground truth
%AF (m x 1) :acquisition function
%posGP (1 x d):probed points
%kGP (n x 1):stiffness at the probed points
%posGP_next (1 x d):points to be probed along the trajectory
%ymu (m x 1) :GP mean
%ys2 (m x 1) :GP uncertainty
%xs_traj ((d+1) x L): trajectory with L ( d-dimensional position and orientation)
%Yifei: only main.m and EvaluateTrajUtility.m has GP related stuff
%--------------------------------------------------------%
opt = [];%initialize options
%%%%%%%%%%%%%%%%Setting domain bounds%%%%%%%%%%%%%%%%%%%%%
DomainBounds.xmin = 0.0;
DomainBounds.xmax = 150.0;
DomainBounds.ymin = 0.0;
DomainBounds.ymax = 150.0;
Lx = DomainBounds.xmax - DomainBounds.xmin;
Ly = DomainBounds.ymax - DomainBounds.ymin;
%Discretizing the domain
xdel=1;
ydel=1;
xr=0:xdel:Lx-xdel;
yr=0:ydel:Ly-ydel;
%--------------------------------------------------------%
%%%%%%%%%%%%%%%%Generate Stiffness Map%%%%%%%%%%%%%%%%%%%%%
addnoise=0;
[X,Y,sGT] = GenerateStiffnessMap(xr,yr,addnoise);
[X,Y,sGT_prior] = GenerateStiffnessMap_prior(xr,yr,addnoise);

sGT = reshape(sGT,size(X));
Z   = zeros(size(X)); %for 3D problem
xss=[X(:),Y(:),Z(:)];
%(3d point, scalar stiffness at the point)
datafull = [xss,sGT(:)];
%--------------------------------------------------------%
%%%%%%%%%%Initialize Prior Acquisition Function%%%%%%%%%%%%%
% opt.gp.AF = zeros(size(X));
opt.gp.AF = sGT_prior;
%--------------------------------------------------------%
%%%%%%%%%%%%%%initialize robot position%%%%%%%%%%%%%%%%%%%%
%Get te index of the position where AF is max
[~,idxinit]=max(opt.gp.AF(:));
idxinit = 6600-70; % any random initial point
%--------------------------------------------------------%
%%%%%%%%%%%%%%%%initialize CE planner%%%%%%%%%%%%%%%%%%%%%%%
[opt] = initialize_gen_traj_CE(opt,DomainBounds);
opt.planner=3;
if(opt.planner==1)
    display('Planner:Utility Maximization')
end

if(opt.planner==2)
    display('Planner:Ergodic Coverage')
end

if(opt.planner==3)
    display('Planner:Deep-RL agent')
end

%If you add a new method, don't forget to update EvaluateUtility.m

opt.method=1;
if(opt.method==1)
    display('Method: Expected improvement')
end

if(opt.method==2)
    display('Method:Variance Reduction')
end

if(opt.method==3)
    display('Method: Level Set Estimation')
end

if(opt.method==4)
    display('Method: Active Area Search')
end

opt.xi = [xss(idxinit,:)'; -0.0*pi];%[pos;orientation]
traj_save=[]; %save trajectories
%--------------------------------------------------------%
%%%%%%%%%%%%%%%%initialize ergodicity%%%%%%%%%%%%%%%%%%%%%%%
if(opt.planner==2)
    opt.erg.mu=opt.gp.AF./sum(sum(opt.gp.AF));
    % Number of wave-numbers to be used
    Nk = 10;%%
    opt.erg.Nkx = Nk;
    opt.erg.Nky = Nk;
    %Undo Fourier reflection
%     opt.erg.mu=flipud(opt.erg.mu);
%     opt.erg.mu=imrotate(opt.erg.mu,-90);
    [opt.erg.muk] = GetFourierCoeff(opt,X,Y);
    opt.erg.Ck = zeros(Nk,Nk);
end
%--------------------------------------------------------%
%%%%%%%%%%%%%%%%initialize GP%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GPRsave.ymusave=[];GPRsave.ys2save=[]; GPRsave.AFsave=[];
opt.gp.posGP=[];opt.gp.kGP=[];sample_rate=5;
%GP parameters
opt.gp_model = struct('inf',@infExact, 'mean', @meanZero, 'cov', @covSEiso, 'lik', @likGauss);
sn = 0.01; ell = 14; sf = sqrt(1);Ncg=30; in_noise = 0;% try 50 to notice differece 
opt.gp_para.lik = log(sn); opt.gp_para.cov = [log([ell; sf])];%in_noise];
opt.gp.posGP=[]; %opt.xi(1:opt.dim,:)';
opt.gp.kGP=[];%EvaluateStiffnessKnn(opt.gp.posGP,xss,sGT);

if(opt.method==3)
    opt.level = 0.5;
    opt.beta_t = 1.96;
    opt.eps_band = .01;
    opt.method_obj = ActiveLevelSetEstimation(opt.gp_model, opt.gp_para, xss, opt.level, opt.beta_t, opt.eps_band);
end

if(opt.method==4)
    opt.level=0.5;
    [region_grid_x, region_grid_y] = meshgrid(0:10:130, 0:10:130);
    opt.regions = [region_grid_x(:), region_grid_x(:)+20, region_grid_y(:), region_grid_y(:)+20];
%     opt.regions(:,5) = -0.1;
%     opt.regions(:,6) = 0.1;
    opt.method_obj = ActiveAreaSearch(opt.gp_model, opt.gp_para, xss(:,1:2), opt.regions, opt.level, 1, .8);
    opt.kdOBJ = KDTreeSearcher(xss(:,1:2));
%     opt.u = opt.method_obj.utility();
% % [idxInfo, ~] = knnsearch(kdOBJ,xstemp(1:opt.dim,:)');
% % f =-sum(opt.info(idxInfo,opt.dim+1));
end

ymu=zeros(size(xss(:,1)));
ys2=zeros(size(xss(:,1)));
GPRsave.ymusave(1,:)=ymu;
GPRsave.ys2save(1,:)=ys2;
GPRsave.AFsave(1,:)=opt.gp.AF(:);


%--------------------------------------------------------%
%%%%%%%%%%%%%%%%initialize figures%%%%%%%%%%%%%%%%%%%%%%%
figure(1);set(gcf,'color','w');
%--------------------------------------------------------%


set(gcf, 'Position', [100, 400, 400, 400]);hold on;
h1=color_line3(xss(:,1), xss(:,2), xss(:,3),ymu,'.');
opt.ceFig_optimal=  draw_path(traj(0*opt.z, opt), 'r', 2, 5);
h1GP=scatter3(0,0,0,20,'filled','mo');
axis equal
axis([ DomainBounds.xmin DomainBounds.xmax DomainBounds.ymin DomainBounds.ymax])
view(0,90)
title('Predicted stiffness map and optimal trajectory')
figure(2);set(gcf,'color','w');
set(gcf, 'Position', [500, 400, 400, 400]);hold on;
h2=color_line3(xss(:,1), xss(:,2), xss(:,3),opt.gp.AF(:),'.') ;
opt.ceFig_candidate= draw_path(rand(3,3), 'b',3, 5);    %plot trajectories inside cem
h2GP=scatter3(0,0,0,20,'filled','mo');
axis equal
axis([ DomainBounds.xmin DomainBounds.xmax DomainBounds.ymin DomainBounds.ymax])
view(0,90)
title('Acquisition function and candidate trajectories')
figure(3);set(gcf,'color','w');
set(gcf, 'Position', [900, 400, 400, 400]);hold on;
color_line3(xss(:,1), xss(:,2), xss(:,3),sGT,'.');
axis equal
axis([ DomainBounds.xmin DomainBounds.xmax DomainBounds.ymin DomainBounds.ymax])
view(0,90)
opt.ceFlag=0;

figure(4);set(gcf,'color','w');
set(gcf, 'Position', [1500, 400, 400, 400]);hold on;
h3 = color_line3(xss(:,1), xss(:,2), xss(:,3),ymu,'.');
title('Predicted stiffness map')
axis equal
axis([ DomainBounds.xmin DomainBounds.xmax DomainBounds.ymin DomainBounds.ymax])

%--------------------------------------------------------%

%% Receding-horizon trajectory planning
traj_save=[];
for k=1:opt.stages %number of iterations
    pause
    opt.currentStage = k
    if(opt.planner==1 || opt.planner==2)
        if k==15
            opt.planner = 1;
            display('Switch Planner ---> Utility Maximization')
        end
        if k == 1
            opt.tf = 20;
        else
            opt.tf = 50;
        end
        [opt,xs_traj] = gen_traj_CE(opt);
    elseif(opt.planner == 3)
        
        if(size(opt.gp.AF,2)==1)
            a = sqrt(length(opt.gp.AF));
            goal_mat = reshape(opt.gp.AF/max(max(opt.gp.AF)),[a,a]);
            goal_mat(goal_mat>0.8)=100;
            goal_mat(goal_mat<0.8)=0;
        else
            goal_mat = opt.gp.AF/max(max(opt.gp.AF));
            goal_mat(goal_mat>0.8)=100;
            goal_mat(goal_mat<0.8)=0;
        end
        goal_mat = imresize(goal_mat,[60, 60]);
        goal_mat = 0.2*flipud(goal_mat);
%         imshow(goal_mat)
        
        % Call deepRL service
        RLclient = rossvcclient('compute_traj');
        msg = rosmessage(RLclient);
        msg.AqFunction.Data=goal_mat(:);
        pose = call(RLclient,msg);
        pose_falattened = pose.PoseAgent.Data;

        y_agent = pose_falattened(1:2:end-1);
        x_agent = pose_falattened(2:2:end);
        
%         plot(x_agent,90-y_agent)
%         axis([0 90 0 90])
        y_agent=-y_agent+90;
        %scale back to original size
        y_agent = (y_agent-12)*150/60;
        x_agent = (x_agent-12)*150/60;
        
        %Maintain same structure as the old trajectory sampling code
        xs_traj = [x_agent,y_agent,zeros(size(x_agent,1),1),zeros(size(x_agent,1),1)]';

    end
    %%
    %Execute the trajectory
    if(opt.planner==1)
        %Utility Maximization: pick the best sample along the trajectory
        [traj_utility] = EvaluateTrajUtility(xs_traj,opt);
        [~,idx_utility]=max(traj_utility);
        posGP_new=xs_traj(1:opt.dim,idx_utility)';
        traj_save=[traj_save,xs_traj(:,1:idx_utility)];
        %update the initial condition for trajectory
        xf=xs_traj(:,idx_utility);
    end
    
    if(opt.planner==2)
        %Ergodic coverage: pick multiple samples with fixed-sampling rate
        posGP_new=xs_traj(1:opt.dim,end)';
        traj_save=[traj_save,xs_traj];
        %update the initial condition for trajectory
        xf=xs_traj(:,end);
        opt = accumulate_CK(opt, xs_traj);% updates CK of the whole trajectory (for efficient calculation)
    end
    
    if(opt.planner==3)
        %DeepRL agent: pick multiple samples with fixed-sampling rate
        posGP_new=xs_traj(1:opt.dim,1:20:end)';
        traj_save=[traj_save,xs_traj];
        %update the initial condition for trajectory
        xf=xs_traj(:,end);
    end
    
    opt.xi = xf;
    
    %GPR update
    opt.gp.posGP=[opt.gp.posGP;posGP_new];
    kGP_new=EvaluateStiffnessKnn(posGP_new,xss,sGT);
    opt.gp.kGP= [opt.gp.kGP; kGP_new];
    %remove points that are too close to each other from the training set
    %[posGPfilt,kGPfilt]=consolidator(posGP,kGP,'max',2);%noise
    [ymu, ys2, fmu, fs2]= gp(opt.gp_para,opt.gp_model.inf, opt.gp_model.mean, opt.gp_model.cov, opt.gp_model.lik, opt.gp.posGP, opt.gp.kGP, xss);
    
    if opt.method==3
        opt.method_obj.update(posGP_new, kGP_new);
    end
    
    if opt.method==4
        opt.method_obj.update(posGP_new(:, 1:2), kGP_new);
    end
    
    if(opt.method==1)%Expected Improvement
        yEI=max(opt.gp.kGP);%current max
        opt.gp.AF = EI(yEI,ymu,ys2);
    end
    if(opt.method==2)%Variance Reduction
        opt.gp.AF = ys2;
    end
    if opt.method>=3
        opt.gp.AF = opt.method_obj.utility();
    end
    
    if(opt.planner==2)%Ergodic coverage
        opt.erg.mu=opt.gp.AF./sum(sum(opt.gp.AF));
        % Number of wave-numbers to be used
%         opt.erg.mu=flipud(opt.erg.mu);
%         opt.erg.mu=imrotate(opt.erg.mu,-90);
        [opt.erg.muk] = GetFourierCoeff(opt,X,Y);
        
    end
    
    set(h1,'XData', [xss(:,1) xss(:,1)],'YData', [xss(:,2) xss(:,2)],'ZData',[xss(:,3) xss(:,3)] ,'CData', [ymu ymu])
    set(h2,'XData', [xss(:,1) xss(:,1)],'YData', [xss(:,2) xss(:,2)],'ZData',[xss(:,3) xss(:,3)] ,'CData', [opt.gp.AF(:) opt.gp.AF(:)])
    set(h3,'XData', [xss(:,1) xss(:,1)],'YData', [xss(:,2) xss(:,2)],'ZData',[xss(:,3) xss(:,3)] ,'CData', [ymu ymu])   
    set(opt.ceFig_optimal,'XData', traj_save(1,:),'YData', traj_save(2,:),'ZData', traj_save(3,:));
    set(h1GP,'XData',opt.gp.posGP(:,1) ,'YData',opt.gp.posGP(:,2) ,'ZData', opt.gp.posGP(:,3));  %sampled points
    set(h2GP,'XData',opt.gp.posGP(:,1) ,'YData',opt.gp.posGP(:,2) ,'ZData', opt.gp.posGP(:,3));  %sampled points
    
    drawnow
    
    GPRsave.ymusave(opt.stages+1,:)=ymu;
    GPRsave.ys2save(opt.stages+1,:)=ys2;
    GPRsave.AFsave(opt.stages+1,:)=opt.gp.AF(:);
    
end
