/**
 * Subagent activity monitor component.
 * 
 * This component displays real-time information about subagent activities,
 * including active subagents, recent communications, and performance metrics.
 */

"use client";

import { useEffect, useState } from 'react';
import { useCopilotAgent } from '@/hooks/useCopilotAgent';
import { 
  Users, 
  Activity, 
  MessageSquare, 
  Clock, 
  CheckCircle, 
  XCircle, 
  AlertTriangle,
  RefreshCw,
  TrendingUp,
  TrendingDown,
  Minus
} from 'lucide-react';
import { Subagent, TaskStatus } from '@/types';

interface SubagentActivity {
  id: string;
  name: string;
  status: 'active' | 'idle' | 'busy' | 'error';
  last_activity: string;
  tasks_completed: number;
  tasks_failed: number;
  avg_response_time: number;
  current_task?: string;
  capabilities: string[];
}

interface SubagentCommunication {
  id: string;
  subagent_name: string;
  message_type: 'request' | 'response' | 'error';
  timestamp: string;
  content_preview: string;
  status: 'success' | 'pending' | 'failed';
}

export function SubagentActivityMonitor() {
  const { getAgentStatus, isLoading } = useCopilotAgent();
  const [subagentActivities, setSubagentActivities] = useState<SubagentActivity[]>([]);
  const [recentCommunications, setRecentCommunications] = useState<SubagentCommunication[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

  const fetchSubagentData = async () => {
    setRefreshing(true);
    try {
      // In a real implementation, this would fetch from dedicated subagent endpoints
      // For now, we'll generate mock data that represents realistic subagent activity
      
      const mockActivities: SubagentActivity[] = [
        {
          id: 'reflection-agent',
          name: 'Reflection Agent',
          status: 'active',
          last_activity: new Date(Date.now() - 30000).toISOString(),
          tasks_completed: 15,
          tasks_failed: 1,
          avg_response_time: 250,
          current_task: 'Analyzing recent conversation patterns',
          capabilities: ['self-analysis', 'pattern-recognition', 'improvement-suggestions']
        },
        {
          id: 'builder-agent',
          name: 'Builder Agent',
          status: 'idle',
          last_activity: new Date(Date.now() - 300000).toISOString(),
          tasks_completed: 8,
          tasks_failed: 0,
          avg_response_time: 450,
          capabilities: ['code-generation', 'capability-creation', 'system-extension']
        },
        {
          id: 'external-input-agent',
          name: 'External Input Agent',
          status: 'busy',
          last_activity: new Date(Date.now() - 5000).toISOString(),
          tasks_completed: 23,
          tasks_failed: 2,
          avg_response_time: 180,
          current_task: 'Processing external API data',
          capabilities: ['api-integration', 'data-processing', 'external-communication']
        }
      ];

      const mockCommunications: SubagentCommunication[] = [
        {
          id: '1',
          subagent_name: 'Reflection Agent',
          message_type: 'response',
          timestamp: new Date(Date.now() - 45000).toISOString(),
          content_preview: 'Analysis complete: Found 3 areas for improvement...',
          status: 'success'
        },
        {
          id: '2',
          subagent_name: 'External Input Agent',
          message_type: 'request',
          timestamp: new Date(Date.now() - 120000).toISOString(),
          content_preview: 'Requesting permission to access external API...',
          status: 'pending'
        },
        {
          id: '3',
          subagent_name: 'Builder Agent',
          message_type: 'response',
          timestamp: new Date(Date.now() - 180000).toISOString(),
          content_preview: 'New capability module created successfully',
          status: 'success'
        },
        {
          id: '4',
          subagent_name: 'Reflection Agent',
          message_type: 'error',
          timestamp: new Date(Date.now() - 240000).toISOString(),
          content_preview: 'Failed to access memory store: Connection timeout',
          status: 'failed'
        }
      ];

      setSubagentActivities(mockActivities);
      setRecentCommunications(mockCommunications);
      setLastRefresh(new Date());
    } catch (error) {
      console.error('Failed to fetch subagent data:', error);
    } finally {
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchSubagentData();
    const interval = setInterval(fetchSubagentData, 10000); // Update every 10 seconds
    return () => clearInterval(interval);
  }, []);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'busy':
        return <Activity className="h-4 w-4 text-blue-500 animate-pulse" />;
      case 'idle':
        return <Clock className="h-4 w-4 text-yellow-500" />;
      case 'error':
        return <XCircle className="h-4 w-4 text-red-500" />;
      default:
        return <Minus className="h-4 w-4 text-gray-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-green-100 text-green-800';
      case 'busy':
        return 'bg-blue-100 text-blue-800';
      case 'idle':
        return 'bg-yellow-100 text-yellow-800';
      case 'error':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getCommunicationIcon = (type: string, status: string) => {
    if (status === 'failed') {
      return <XCircle className="h-4 w-4 text-red-500" />;
    }
    if (status === 'pending') {
      return <Clock className="h-4 w-4 text-yellow-500" />;
    }
    
    switch (type) {
      case 'request':
        return <TrendingUp className="h-4 w-4 text-blue-500" />;
      case 'response':
        return <TrendingDown className="h-4 w-4 text-green-500" />;
      case 'error':
        return <AlertTriangle className="h-4 w-4 text-red-500" />;
      default:
        return <MessageSquare className="h-4 w-4 text-gray-500" />;
    }
  };

  const formatTimeAgo = (timestamp: string) => {
    const now = new Date();
    const time = new Date(timestamp);
    const diff = now.getTime() - time.getTime();
    
    if (diff < 60000) {
      return `${Math.floor(diff / 1000)}s ago`;
    } else if (diff < 3600000) {
      return `${Math.floor(diff / 60000)}m ago`;
    } else {
      return time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Users className="h-6 w-6 text-blue-600" />
          <h2 className="text-xl font-bold text-gray-900">Subagent Activity</h2>
        </div>
        <button
          onClick={fetchSubagentData}
          disabled={refreshing}
          className="flex items-center space-x-2 px-3 py-1 text-sm bg-blue-50 text-blue-600 rounded-md hover:bg-blue-100 disabled:opacity-50"
        >
          <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Active Subagents */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Active Subagents</h3>
        
        <div className="space-y-4">
          {subagentActivities.map((activity) => (
            <div key={activity.id} className="border border-gray-200 rounded-lg p-4">
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center space-x-3">
                  {getStatusIcon(activity.status)}
                  <div>
                    <h4 className="font-medium text-gray-900">{activity.name}</h4>
                    <span className={`inline-block px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(activity.status)}`}>
                      {activity.status.charAt(0).toUpperCase() + activity.status.slice(1)}
                    </span>
                  </div>
                </div>
                <div className="text-right text-sm text-gray-500">
                  <div>Last active: {formatTimeAgo(activity.last_activity)}</div>
                  <div>Avg response: {activity.avg_response_time}ms</div>
                </div>
              </div>

              {activity.current_task && (
                <div className="mb-3 p-2 bg-blue-50 rounded-md">
                  <div className="flex items-center space-x-2">
                    <Activity className="h-4 w-4 text-blue-500" />
                    <span className="text-sm text-blue-700 font-medium">Current Task:</span>
                  </div>
                  <p className="text-sm text-blue-600 mt-1">{activity.current_task}</p>
                </div>
              )}

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-3">
                <div className="text-center p-2 bg-green-50 rounded-md">
                  <div className="text-lg font-bold text-green-600">{activity.tasks_completed}</div>
                  <div className="text-xs text-gray-600">Completed</div>
                </div>
                <div className="text-center p-2 bg-red-50 rounded-md">
                  <div className="text-lg font-bold text-red-600">{activity.tasks_failed}</div>
                  <div className="text-xs text-gray-600">Failed</div>
                </div>
                <div className="text-center p-2 bg-blue-50 rounded-md">
                  <div className="text-lg font-bold text-blue-600">
                    {((activity.tasks_completed / (activity.tasks_completed + activity.tasks_failed)) * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-600">Success Rate</div>
                </div>
              </div>

              <div>
                <div className="text-sm text-gray-600 mb-2">Capabilities:</div>
                <div className="flex flex-wrap gap-1">
                  {activity.capabilities.map((capability, index) => (
                    <span
                      key={index}
                      className="inline-block px-2 py-1 text-xs bg-gray-100 text-gray-700 rounded-md"
                    >
                      {capability}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Recent Communications */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Communications</h3>
        
        <div className="space-y-3">
          {recentCommunications.map((comm) => (
            <div key={comm.id} className="flex items-start space-x-3 p-3 border border-gray-200 rounded-lg">
              {getCommunicationIcon(comm.message_type, comm.status)}
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <span className="font-medium text-gray-900">{comm.subagent_name}</span>
                    <span className="text-sm text-gray-500">•</span>
                    <span className="text-sm text-gray-500 capitalize">{comm.message_type}</span>
                  </div>
                  <span className="text-sm text-gray-500">{formatTimeAgo(comm.timestamp)}</span>
                </div>
                <p className="text-sm text-gray-600 mt-1 truncate">{comm.content_preview}</p>
                <div className="flex items-center mt-2">
                  <span className={`inline-block w-2 h-2 rounded-full mr-2 ${
                    comm.status === 'success' ? 'bg-green-500' :
                    comm.status === 'pending' ? 'bg-yellow-500' : 'bg-red-500'
                  }`} />
                  <span className="text-xs text-gray-500 capitalize">{comm.status}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="text-sm text-gray-500 text-center">
        Last updated: {lastRefresh.toLocaleTimeString()}
      </div>
    </div>
  );
}