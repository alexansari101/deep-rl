%% Initialization
clc;close all;clear;
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
sGT=reshape(sGT,size(X));
Z=zeros(size(X)); %for 3D problem
xss=[X(:),Y(:),Z(:)];
%(3d point, scalar stiffness at the point)
datafull=[xss,sGT(:)];
%--------------------------------------------------------%
%%%%%%%%%%Initialize Prior Acquisition Function%%%%%%%%%%%%%
opt.gp.AF=zeros(size(X));
%--------------------------------------------------------%
%%%%%%%%%%%%%%initialize robot position%%%%%%%%%%%%%%%%%%%%
%Get te index of the position where AF is max
[~,idxinit]=max(opt.gp.AF(:));
%--------------------------------------------------------%
%%%%%%%%%%%%%%%%initialize CE planner%%%%%%%%%%%%%%%%%%%%%%%
[opt] = initialize_gen_traj_CE(opt,DomainBounds);
opt.planner=1;
if(opt.planner==1)
    display('Planner:Utility Maximization')
end

if(opt.planner==2)
    display('Planner:Ergodic Coverage')
end

%If you add a new method, don't forget to update EvaluateUtility.m

opt.method=1;
if(opt.method==1)
    display('Method: Expected improvement')
end

if(opt.method==2)
    display('Method:Variance Reduction')
end

opt.xi = [xss(idxinit,:)'; 0.1*pi];%[pos;orientation]
traj_save=opt.xi; %save trajectories
%--------------------------------------------------------%
%% %%%%%%%%%%%%%%%initialize ergodicity%%%%%%%%%%%%%%%%%%%%%%%
if(opt.planner==2)
    opt.erg.mu=opt.gp.AF./sum(sum(opt.gp.AF));
    % Number of wave-numbers to be used
    Nk = 10;%%
    opt.erg.Nkx = Nk;
    opt.erg.Nky = Nk;
    %Undo Fourier reflection
    opt.erg.mu=flipud(opt.erg.mu);
    opt.erg.mu=imrotate(opt.erg.mu,-90);
    [opt.erg.muk] = GetFourierCoeff(opt,X,Y);
end
%--------------------------------------------------------%
%% %%%%%%%%%%%%%%%initialize GP%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GPRsave.ymusave=[];GPRsave.ys2save=[]; GPRsave.AFsave=[];
opt.gp.posGP=[];opt.gp.kGP=[];sample_rate=5;
%GP parameters
opt.gp_model = struct('inf',@infExact, 'mean', @meanZero, 'cov', @covSEiso, 'lik', @likGauss);
sn = 0.01; ell =8; sf = sqrt(1);Ncg=30;
opt.gp_para.lik = log(sn); opt.gp_para.cov = log([ell; sf]);
opt.gp.posGP=opt.xi(1:opt.dim,:)';
opt.gp.kGP=EvaluateStiffnessKnn(opt.gp.posGP,xss,sGT);
ymu=zeros(size(xss(:,1)));
ys2=zeros(size(xss(:,1)));
GPRsave.ymusave(1,:)=ymu;
GPRsave.ys2save(1,:)=ys2;
GPRsave.AFsave(1,:)=opt.gp.AF(:);

%--------------------------------------------------------%
%% %%%%%%%%%%%%%%%initialize figures%%%%%%%%%%%%%%%%%%%%%%%
figure(1);set(gcf,'color','w');
set(gcf, 'Position', [100, 400, 400, 400]);hold on;
h1=color_line3(xss(:,1), xss(:,2), xss(:,3),ymu,'.');
opt.ceFig_optimal=  draw_path(traj(opt.z, opt), 'r', 2, 5);
h1GP=scatter3(opt.gp.posGP(:,1),opt.gp.posGP(:,2),opt.gp.posGP(:,3),20,'filled','mo');
axis equal
axis([ DomainBounds.xmin DomainBounds.xmax DomainBounds.ymin DomainBounds.ymax])
view(0,90)
title('Predicted stiffness map and optimal trajectory')
figure(2);set(gcf,'color','w');
set(gcf, 'Position', [500, 400, 400, 400]);hold on;
h2=color_line3(xss(:,1), xss(:,2), xss(:,3),opt.gp.AF(:),'.') ;
opt.ceFig_candidate= draw_path(rand(3,3), 'b',3, 5);    %plot trajectories inside cem
h2GP=scatter3(opt.gp.posGP(:,1),opt.gp.posGP(:,2),opt.gp.posGP(:,3),20,'filled','mo');
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
%--------------------------------------------------------%

%% Receding-horizon trajectory planning

for k=1:opt.stages %number of iterations
    
    [opt,xs_traj] = gen_traj_CE(opt);
    
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
        posGP_new=xs_traj(1:opt.dim,2:4:end)';
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
    yEI=max(opt.gp.kGP);%current max
    
    if(opt.method==1)%Expected Improvement
        opt.gp.AF = EI(yEI,ymu,ys2);
    end
    if(opt.method==2)%Variance Reduction
        opt.gp.AF = ys2;
    end
    
    if(opt.planner==2)%Ergodic coverage
        opt.erg.mu=opt.gp.AF./sum(sum(opt.gp.AF));
        % Number of wave-numbers to be used
        opt.erg.mu=flipud(opt.erg.mu);
        opt.erg.mu=imrotate(opt.erg.mu,-90);
        [opt.erg.muk] = GetFourierCoeff(opt,X,Y);
    end
    
    
    set(h1,'XData', [xss(:,1) xss(:,1)],'YData', [xss(:,2) xss(:,2)],'ZData',[xss(:,3) xss(:,3)] ,'CData', [ymu ymu])
    set(h2,'XData', [xss(:,1) xss(:,1)],'YData', [xss(:,2) xss(:,2)],'ZData',[xss(:,3) xss(:,3)] ,'CData', [opt.gp.AF(:) opt.gp.AF(:)])
    set(opt.ceFig_optimal,'XData', traj_save(1,:),'YData', traj_save(2,:),'ZData', traj_save(3,:));
    set(h1GP,'XData',opt.gp.posGP(:,1) ,'YData',opt.gp.posGP(:,2) ,'ZData', opt.gp.posGP(:,3));  %sampled points
    set(h2GP,'XData',opt.gp.posGP(:,1) ,'YData',opt.gp.posGP(:,2) ,'ZData', opt.gp.posGP(:,3));  %sampled points
    
    drawnow
    
    GPRsave.ymusave(opt.stages+1,:)=ymu;
    GPRsave.ys2save(opt.stages+1,:)=ys2;
    GPRsave.AFsave(opt.stages+1,:)=opt.gp.AF(:);
    
end
