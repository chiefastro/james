/**
 * Comprehensive agent status dashboard component.
 * 
 * This component provides detailed monitoring of James' status including:
 * - Active task display and management
 * - Memory and subagent activity monitors
 * - System health indicators and metrics
 * - Real-time updates with responsive design
 */

"use client";

import { useEffect, useState } from 'react';
import { useCopilotAgent } from '@/hooks/useCopilotAgent';
import { useAgentConnection } from '@/hooks/useAgentConnection';
import { 
  Activity, 
  Brain, 
  MessageSquare, 
  Users, 
  Clock, 
  AlertCircle,
  CheckCircle,
  XCircle,
  Pause,
  Play,
  BarChart3,
  Database,
  Cpu,
  HardDrive,
  Wifi,
  RefreshCw,
  TrendingUp,
  TrendingDown,
  Minus
} from 'lucide-react';
import { AgentStatusResponse, Task, Subagent, TaskStatus } from '@/types';

interface MemoryMetrics {
  episodic_count: number;
  semantic_count: number;
  procedural_count: number;
  working_memory_size: number;
  total_size_mb: number;
  cleanup_last_run?: string;
}

interface SystemMetrics {
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  network_latency: number;
  error_rate: number;
  response_time_avg: number;
}

export function AgentDashboard() {
  const { getAgentStatus, isLoading, error } = useCopilotAgent();
  const { isConnected, agentState } = useAgentConnection();
  const [status, setStatus] = useState<AgentStatusResponse | null>(null);
  const [memoryMetrics, setMemoryMetrics] = useState<MemoryMetrics | null>(null);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

  const fetchDashboardData = async () => {
    setRefreshing(true);
    try {
      const agentStatus = await getAgentStatus();
      if (agentStatus) {
        setStatus(agentStatus);
        // Extract memory metrics from status if available
        if (agentStatus.memory_usage) {
          setMemoryMetrics(agentStatus.memory_usage as MemoryMetrics);
        }
        // Mock system metrics for now - in real implementation these would come from backend
        setSystemMetrics({
          cpu_usage: Math.random() * 100,
          memory_usage: Math.random() * 100,
          disk_usage: Math.random() * 100,
          network_latency: Math.random() * 100,
          error_rate: Math.random() * 5,
          response_time_avg: Math.random() * 1000
        });
      }
      setLastRefresh(new Date());
    } catch (err) {
      console.error('Failed to fetch dashboard data:', err);
    } finally {
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 15000); // Update every 15 seconds
    return () => clearInterval(interval);
  }, [getAgentStatus]);

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours}h ${minutes}m ${secs}s`;
  };

  const getTaskStatusIcon = (status: TaskStatus) => {
    switch (status) {
      case TaskStatus.COMPLETED:
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case TaskStatus.FAILED:
        return <XCircle className="h-4 w-4 text-red-500" />;
      case TaskStatus.IN_PROGRESS:
        return <Play className="h-4 w-4 text-blue-500" />;
      case TaskStatus.PENDING:
        return <Pause className="h-4 w-4 text-yellow-500" />;
      case TaskStatus.CANCELLED:
        return <Minus className="h-4 w-4 text-gray-500" />;
      default:
        return <Clock className="h-4 w-4 text-gray-400" />;
    }
  };

  const getHealthColor = (value: number, threshold: { good: number; warning: number }) => {
    if (value <= threshold.good) return 'text-green-600 bg-green-100';
    if (value <= threshold.warning) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getTrendIcon = (current: number, previous: number) => {
    if (current > previous) return <TrendingUp className="h-3 w-3 text-red-500" />;
    if (current < previous) return <TrendingDown className="h-3 w-3 text-green-500" />;
    return <Minus className="h-3 w-3 text-gray-400" />;
  };

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <AlertCircle className="h-5 w-5 text-red-500 mr-2" />
            <span className="text-red-700 font-medium">Dashboard Error</span>
          </div>
          <button
            onClick={fetchDashboardData}
            className="text-red-600 hover:text-red-800 text-sm underline"
          >
            Retry
          </button>
        </div>
        <p className="text-red-600 text-sm mt-2">Failed to load agent dashboard data</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Dashboard Header */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <Brain className="h-6 w-6 text-blue-600" />
            <h2 className="text-xl font-bold text-gray-900">James Dashboard</h2>
          </div>
          <div className="flex items-center space-x-4">
            <div className={`flex items-center space-x-2 ${isConnected ? 'text-green-600' : 'text-red-600'}`}>
              <Wifi className="h-4 w-4" />
              <span className="text-sm font-medium">
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            <button
              onClick={fetchDashboardData}
              disabled={refreshing}
              className="flex items-center space-x-2 px-3 py-1 text-sm bg-blue-50 text-blue-600 rounded-md hover:bg-blue-100 disabled:opacity-50"
            >
              <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
              <span>Refresh</span>
            </button>
          </div>
        </div>
        
        <div className="text-sm text-gray-500">
          Last updated: {lastRefresh.toLocaleTimeString()}
        </div>
      </div>

      {/* Status Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow-md p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Agent Status</p>
              <p className="text-lg font-semibold">
                {status?.is_active ? 'Active' : 'Inactive'}
              </p>
            </div>
            <div className={`p-2 rounded-full ${status?.is_active ? 'bg-green-100' : 'bg-gray-100'}`}>
              <Brain className={`h-5 w-5 ${status?.is_active ? 'text-green-600' : 'text-gray-400'}`} />
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Message Queue</p>
              <p className="text-lg font-semibold">
                {status?.message_queue_size || agentState.queueSize || 0}
              </p>
            </div>
            <div className="p-2 rounded-full bg-blue-100">
              <MessageSquare className="h-5 w-5 text-blue-600" />
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Active Tasks</p>
              <p className="text-lg font-semibold">
                {status?.current_tasks?.length || agentState.currentTasks.length || 0}
              </p>
            </div>
            <div className="p-2 rounded-full bg-purple-100">
              <Activity className="h-5 w-5 text-purple-600" />
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Active Subagents</p>
              <p className="text-lg font-semibold">
                {status?.active_subagents || 0}
              </p>
            </div>
            <div className="p-2 rounded-full bg-orange-100">
              <Users className="h-5 w-5 text-orange-600" />
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Active Tasks Panel */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Active Tasks</h3>
            <Activity className="h-5 w-5 text-purple-600" />
          </div>
          
          <div className="space-y-3">
            {status?.current_tasks && status.current_tasks.length > 0 ? (
              status.current_tasks.slice(0, 5).map((task: Task) => (
                <div key={task.id} className="flex items-start space-x-3 p-3 bg-gray-50 rounded-lg">
                  {getTaskStatusIcon(task.status)}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      {task.description}
                    </p>
                    <div className="flex items-center space-x-2 mt-1">
                      <span className="text-xs text-gray-500">
                        Priority: {task.priority}
                      </span>
                      <span className="text-xs text-gray-500">
                        {new Date(task.created_at).toLocaleTimeString()}
                      </span>
                    </div>
                    {task.assigned_subagents.length > 0 && (
                      <div className="flex items-center space-x-1 mt-1">
                        <Users className="h-3 w-3 text-gray-400" />
                        <span className="text-xs text-gray-500">
                          {task.assigned_subagents.length} subagent(s)
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-8 text-gray-500">
                <Activity className="h-8 w-8 mx-auto mb-2 text-gray-300" />
                <p>No active tasks</p>
              </div>
            )}
          </div>
        </div>

        {/* Memory Metrics Panel */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Memory Usage</h3>
            <Database className="h-5 w-5 text-green-600" />
          </div>
          
          {memoryMetrics ? (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-3 bg-blue-50 rounded-lg">
                  <p className="text-2xl font-bold text-blue-600">{memoryMetrics.episodic_count}</p>
                  <p className="text-sm text-gray-600">Episodic</p>
                </div>
                <div className="text-center p-3 bg-green-50 rounded-lg">
                  <p className="text-2xl font-bold text-green-600">{memoryMetrics.semantic_count}</p>
                  <p className="text-sm text-gray-600">Semantic</p>
                </div>
                <div className="text-center p-3 bg-purple-50 rounded-lg">
                  <p className="text-2xl font-bold text-purple-600">{memoryMetrics.procedural_count}</p>
                  <p className="text-sm text-gray-600">Procedural</p>
                </div>
                <div className="text-center p-3 bg-orange-50 rounded-lg">
                  <p className="text-2xl font-bold text-orange-600">{memoryMetrics.working_memory_size}</p>
                  <p className="text-sm text-gray-600">Working</p>
                </div>
              </div>
              
              <div className="pt-3 border-t border-gray-200">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Total Size</span>
                  <span className="font-medium">{memoryMetrics.total_size_mb.toFixed(1)} MB</span>
                </div>
                {memoryMetrics.cleanup_last_run && (
                  <div className="flex justify-between items-center mt-1">
                    <span className="text-sm text-gray-600">Last Cleanup</span>
                    <span className="text-sm text-gray-500">
                      {new Date(memoryMetrics.cleanup_last_run).toLocaleDateString()}
                    </span>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              <Database className="h-8 w-8 mx-auto mb-2 text-gray-300" />
              <p>Memory metrics unavailable</p>
            </div>
          )}
        </div>
      </div>

      {/* System Health Metrics */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">System Health</h3>
          <BarChart3 className="h-5 w-5 text-indigo-600" />
        </div>
        
        {systemMetrics ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Cpu className="h-4 w-4 text-blue-500" />
                  <span className="text-sm text-gray-600">CPU Usage</span>
                </div>
                <span className={`text-sm font-medium px-2 py-1 rounded ${getHealthColor(systemMetrics.cpu_usage, { good: 50, warning: 80 })}`}>
                  {systemMetrics.cpu_usage.toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${Math.min(systemMetrics.cpu_usage, 100)}%` }}
                />
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <HardDrive className="h-4 w-4 text-green-500" />
                  <span className="text-sm text-gray-600">Memory Usage</span>
                </div>
                <span className={`text-sm font-medium px-2 py-1 rounded ${getHealthColor(systemMetrics.memory_usage, { good: 60, warning: 85 })}`}>
                  {systemMetrics.memory_usage.toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-green-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${Math.min(systemMetrics.memory_usage, 100)}%` }}
                />
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Wifi className="h-4 w-4 text-purple-500" />
                  <span className="text-sm text-gray-600">Network Latency</span>
                </div>
                <span className={`text-sm font-medium px-2 py-1 rounded ${getHealthColor(systemMetrics.network_latency, { good: 50, warning: 100 })}`}>
                  {systemMetrics.network_latency.toFixed(0)}ms
                </span>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <AlertCircle className="h-4 w-4 text-red-500" />
                  <span className="text-sm text-gray-600">Error Rate</span>
                </div>
                <span className={`text-sm font-medium px-2 py-1 rounded ${getHealthColor(systemMetrics.error_rate, { good: 1, warning: 3 })}`}>
                  {systemMetrics.error_rate.toFixed(2)}%
                </span>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Clock className="h-4 w-4 text-orange-500" />
                  <span className="text-sm text-gray-600">Avg Response</span>
                </div>
                <span className={`text-sm font-medium px-2 py-1 rounded ${getHealthColor(systemMetrics.response_time_avg, { good: 200, warning: 500 })}`}>
                  {systemMetrics.response_time_avg.toFixed(0)}ms
                </span>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Clock className="h-4 w-4 text-gray-500" />
                  <span className="text-sm text-gray-600">Uptime</span>
                </div>
                <span className="text-sm font-medium text-gray-700">
                  {status?.uptime_seconds ? formatUptime(status.uptime_seconds) : 'N/A'}
                </span>
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <BarChart3 className="h-8 w-8 mx-auto mb-2 text-gray-300" />
            <p>System metrics unavailable</p>
          </div>
        )}
      </div>

      {/* Agent Activity Indicator */}
      {agentState.isTyping && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-center space-x-3">
            <div className="flex space-x-1">
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" />
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
            </div>
            <span className="text-blue-700 font-medium">James is actively processing...</span>
          </div>
        </div>
      )}

      {/* Loading Indicator */}
      {isLoading && (
        <div className="text-center py-4">
          <div className="inline-block animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600" />
          <p className="text-sm text-gray-600 mt-2">Loading dashboard data...</p>
        </div>
      )}
    </div>
  );
}