function[GPRsave,ymu,ExpInfo,optGprior]= GPR_BO(x,y,xs,optGprior,GPRsave,jj)
%search coefficients
coeffD=0;coeffP=0.5;np=0.99^jj;
%GP parameters
covfunc = @covSEiso; likfunc = @likGauss;
sn = 0; ell =3; sf = sqrt(1);Ncg=30;
hyp.lik = log(sn); hyp.cov = log([ell; sf]);
[xfilt,yfilt]=consolidator(x,y,'max',2);%noise
% hyp = minimize(hyp,@gp, -Ncg, @infExact,[], covfunc, likfunc, xfilt, yfilt); % opt hypers
% display(exp(hyp.cov));
[ymu, ys2,~,~]= gp(hyp, @infExact, [], covfunc, likfunc, xfilt, yfilt, xs);
ymu(ymu<0)=0;
yEI=max(y);
ExpInfo = EI(yEI,ymu,ys2);
%ExpInfo= UCB(ymu,ys2,1);
% ExpInfo=ExpInfo./max(ExpInfo);
%Add distance penalty
xLast=x(end,1:3);
dist=ipdm(xLast,xs);
%Normalize distances
dist=dist./max(dist);
dist=1-dist;%give higher weightage to close points
% %Reduce Prior
% PriorDist1=sqrt((xLast(1)-30)^2+(xLast(2)-30)^2);%center of prior 1
% PriorDist2=sqrt((xLast(1)-70)^2+(xLast(2)-70)^2);%center of prior 2
% 
% if PriorDist1<30
%     optGprior(:,1)=np* optGprior(:,1);    
% end
% if  PriorDist2<30
%     optGprior(:,2)=np* optGprior(:,2);    
% end
ExpInfo=ExpInfo+coeffD*dist'+np*coeffP*optGprior;
GPRsave.ymusave(jj,:)=ymu;
GPRsave.ys2save(jj,:)=ys2;
GPRsave.EIsave(jj,:)=ExpInfo;
end