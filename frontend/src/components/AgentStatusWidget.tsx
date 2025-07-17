/**
 * Agent status widget component for displaying James' current state.
 * 
 * This component shows real-time information about the conscious agent
 * and integrates with CopilotKit for context-aware suggestions.
 */

"use client";

import { useEffect, useState } from 'react';
import { useCopilotAgent } from '@/hooks/useCopilotAgent';
import { useAgentConnection } from '@/hooks/useAgentConnection';
import { Activity, Brain, MessageSquare, Users, Clock, AlertCircle } from 'lucide-react';
import { AgentStatusResponse } from '@/types';

export function AgentStatusWidget() {
  const { getAgentStatus, isLoading, error } = useCopilotAgent();
  const { isConnected, agentState } = useAgentConnection();
  const [status, setStatus] = useState<AgentStatusResponse | null>(null);

  useEffect(() => {
    const fetchStatus = async () => {
      const agentStatus = await getAgentStatus();
      if (agentStatus) {
        setStatus(agentStatus);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 10000); // Update every 10 seconds

    return () => clearInterval(interval);
  }, [getAgentStatus]);

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <div className="flex items-center">
          <AlertCircle className="h-5 w-5 text-red-500 mr-2" />
          <span className="text-red-700">Failed to load agent status</span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900">James Status</h3>
        <div className={`flex items-center space-x-2 ${isConnected ? 'text-green-600' : 'text-red-600'}`}>
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
          <span className="text-sm font-medium">
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="flex items-center space-x-3">
          <Brain className="h-5 w-5 text-blue-500" />
          <div>
            <p className="text-sm text-gray-600">Status</p>
            <p className="font-medium">
              {status?.is_active ? 'Active' : 'Inactive'}
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          <MessageSquare className="h-5 w-5 text-green-500" />
          <div>
            <p className="text-sm text-gray-600">Queue Size</p>
            <p className="font-medium">
              {status?.message_queue_size || agentState.queueSize || 0}
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          <Activity className="h-5 w-5 text-purple-500" />
          <div>
            <p className="text-sm text-gray-600">Active Tasks</p>
            <p className="font-medium">
              {status?.current_tasks?.length || agentState.currentTasks.length || 0}
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          <Users className="h-5 w-5 text-orange-500" />
          <div>
            <p className="text-sm text-gray-600">Subagents</p>
            <p className="font-medium">
              {status?.active_subagents || 0}
            </p>
          </div>
        </div>
      </div>

      {status?.uptime_seconds && (
        <div className="flex items-center space-x-3 pt-2 border-t border-gray-200">
          <Clock className="h-4 w-4 text-gray-500" />
          <div>
            <p className="text-sm text-gray-600">
              Uptime: {Math.floor(status.uptime_seconds / 3600)}h {Math.floor((status.uptime_seconds % 3600) / 60)}m
            </p>
          </div>
        </div>
      )}

      {agentState.isTyping && (
        <div className="flex items-center space-x-2 text-blue-600">
          <div className="flex space-x-1">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" />
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
          </div>
          <span className="text-sm">James is thinking...</span>
        </div>
      )}

      {isLoading && (
        <div className="text-center py-2">
          <div className="inline-block animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600" />
        </div>
      )}
    </div>
  );
}