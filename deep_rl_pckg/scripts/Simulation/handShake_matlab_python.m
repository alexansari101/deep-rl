
while(true)
if(~robotics.ros.internal.Global.isNodeActive)
    rosinit()
end
%%
%%%
load('goal_to_python.mat')
%%
a = sqrt(length(goal));

goal_mat = reshape(goal,[a,a]);
goal_mat = imresize(goal_mat,[60, 60]);
goal_mat = flipud(goal_mat);
imshow(goal_mat)

% %%%
% 
% msg = rosmessage('std_msgs/Float32MultiArray');
% msg.Data = goal_mat(:);
% mat_pub = rospublisher('/matrix',msg);
% 
% msg.Data = [];
% mat_pub = rospublisher('/matrix',msg);
% 
% 
% pose_sub = rossubscriber('/pose','std_msgs/Float32MultiArray');
% 
% msg = receive(pose_sub);
% 
% pose_flattened=msg.Data;
% 
% x = pose_falttened(1:2:end-1);
% y = pose_falttened(2:2:end);
% 
% plot(x,y)
% 
% display('.....Sending Aquisition Function Message....');
% pause(4)

% send(mat_pub,msg);
traj = []
%% service client 
RLclient = rossvcclient('compute_traj');

msg = rosmessage(RLclient);
msg.AqFunction.Data=goal_mat(:);
pose = call(RLclient,msg)
pose_falattened = pose.PoseAgent.Data;
y = pose_falattened(1:2:end-1);
x = pose_falattened(2:2:end);
traj = [traj;x ];
plot(x,90-y)
axis([0 90 0 90])
% rosgenmsg('deep-rl')
end
