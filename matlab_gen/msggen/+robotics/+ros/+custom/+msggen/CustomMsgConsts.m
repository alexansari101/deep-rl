classdef CustomMsgConsts
    %CustomMsgConsts This class stores all message types
    %   The message types are constant properties, which in turn resolve
    %   to the strings of the actual types.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties (Constant)
        deep_rl_pckg_AddTwoInts = 'deep_rl_pckg/AddTwoInts'
        deep_rl_pckg_AddTwoIntsRequest = 'deep_rl_pckg/AddTwoIntsRequest'
        deep_rl_pckg_AddTwoIntsResponse = 'deep_rl_pckg/AddTwoIntsResponse'
        deep_rl_pckg_Num = 'deep_rl_pckg/Num'
        deep_rl_pckg_computeTraj = 'deep_rl_pckg/computeTraj'
        deep_rl_pckg_computeTrajRequest = 'deep_rl_pckg/computeTrajRequest'
        deep_rl_pckg_computeTrajResponse = 'deep_rl_pckg/computeTrajResponse'
    end
    
    methods (Static, Hidden)
        function messageList = getMessageList
            %getMessageList Generate a cell array with all message types.
            %   The list will be sorted alphabetically.
            
            persistent msgList
            if isempty(msgList)
                msgList = cell(7, 1);
                msgList{1} = 'deep_rl_pckg/AddTwoInts';
                msgList{2} = 'deep_rl_pckg/AddTwoIntsRequest';
                msgList{3} = 'deep_rl_pckg/AddTwoIntsResponse';
                msgList{4} = 'deep_rl_pckg/Num';
                msgList{5} = 'deep_rl_pckg/computeTraj';
                msgList{6} = 'deep_rl_pckg/computeTrajRequest';
                msgList{7} = 'deep_rl_pckg/computeTrajResponse';
            end
            
            messageList = msgList;
        end
        
        function serviceList = getServiceList
            %getServiceList Generate a cell array with all service types.
            %   The list will be sorted alphabetically.
            
            persistent svcList
            if isempty(svcList)
                svcList = cell(2, 1);
                svcList{1} = 'deep_rl_pckg/AddTwoInts';
                svcList{2} = 'deep_rl_pckg/computeTraj';
            end
            
            % The message list was already sorted, so don't need to sort
            % again.
            serviceList = svcList;
        end
    end
end
