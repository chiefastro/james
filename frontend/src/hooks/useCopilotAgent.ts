/**
 * Custom hook for CopilotKit integration with the conscious agent system.
 * 
 * This hook provides CopilotKit-specific functionality for interacting with
 * the conscious agent, including function calling and context management.
 */

import { useCopilotAction, useCopilotReadable } from "@copilotkit/react-core";
import { useCallback, useEffect, useState } from 'react';
import { API_BASE_URL } from '@/lib/copilot-config';
import { 
  AgentStatusResponse, 
  TaskResponse, 
  SubagentResponse, 
  MessageRequest,
  MessageSource,
  ApiResponse 
} from '@/types';

interface UseCopilotAgentReturn {
  sendMessageToAgent: (content: string, priority?: number) => Promise<void>;
  getAgentStatus: () => Promise<AgentStatusResponse | null>;
  listActiveTasks: () => Promise<TaskResponse[]>;
  getSubagents: () => Promise<SubagentResponse[]>;
  isLoading: boolean;
  error: string | null;
}

export function useCopilotAgent(): UseCopilotAgentReturn {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [agentStatus, setAgentStatus] = useState<AgentStatusResponse | null>(null);
  const [activeTasks, setActiveTasks] = useState<TaskResponse[]>([]);
  const [subagents, setSubagents] = useState<SubagentResponse[]>([]);

  // Make agent status readable by CopilotKit
  useCopilotReadable({
    description: "Current status of the conscious agent James",
    value: agentStatus ? {
      isActive: agentStatus.is_active,
      currentTasks: agentStatus.current_tasks?.length || 0,
      queueSize: agentStatus.message_queue_size,
      activeSubagents: agentStatus.active_subagents,
      uptime: agentStatus.uptime_seconds,
      lastActivity: agentStatus.last_activity
    } : null,
  });

  // Make active tasks readable by CopilotKit
  useCopilotReadable({
    description: "Currently active tasks in the conscious agent system",
    value: activeTasks.map(task => ({
      id: task.id,
      description: task.description,
      status: task.status,
      priority: task.priority,
      assignedSubagents: task.assigned_subagents.length
    })),
  });

  // Make subagents readable by CopilotKit
  useCopilotReadable({
    description: "Available subagents in the conscious agent system",
    value: subagents.map(subagent => ({
      name: subagent.name,
      description: subagent.description,
      capabilities: subagent.capabilities,
      isActive: subagent.is_active
    })),
  });

  // API helper function
  const apiCall = useCallback(async <T>(
    endpoint: string, 
    options: RequestInit = {}
  ): Promise<T> => {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      throw new Error(`API call failed: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }, []);

  // Send message to agent
  const sendMessageToAgent = useCallback(async (content: string, priority: number = 5): Promise<void> => {
    setIsLoading(true);
    setError(null);

    try {
      const messageRequest: MessageRequest = {
        content,
        priority,
        source: MessageSource.USER,
        metadata: {
          timestamp: new Date().toISOString(),
          frontend_source: 'copilot'
        }
      };

      await apiCall('/message', {
        method: 'POST',
        body: JSON.stringify(messageRequest),
      });

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to send message';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [apiCall]);

  // Get agent status
  const getAgentStatus = useCallback(async (): Promise<AgentStatusResponse | null> => {
    setIsLoading(true);
    setError(null);

    try {
      const status = await apiCall<AgentStatusResponse>('/agent/status');
      setAgentStatus(status);
      return status;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to get agent status';
      setError(errorMessage);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [apiCall]);

  // List active tasks
  const listActiveTasks = useCallback(async (): Promise<TaskResponse[]> => {
    setIsLoading(true);
    setError(null);

    try {
      const tasks = await apiCall<TaskResponse[]>('/tasks');
      setActiveTasks(tasks);
      return tasks;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to list tasks';
      setError(errorMessage);
      return [];
    } finally {
      setIsLoading(false);
    }
  }, [apiCall]);

  // Get subagents
  const getSubagents = useCallback(async (): Promise<SubagentResponse[]> => {
    setIsLoading(true);
    setError(null);

    try {
      const agents = await apiCall<SubagentResponse[]>('/subagents');
      setSubagents(agents);
      return agents;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to get subagents';
      setError(errorMessage);
      return [];
    } finally {
      setIsLoading(false);
    }
  }, [apiCall]);

  // Register CopilotKit actions
  useCopilotAction({
    name: "send_message",
    description: "Send a message to the conscious agent James",
    parameters: [
      {
        name: "content",
        type: "string",
        description: "The message content to send to James",
        required: true,
      },
      {
        name: "priority",
        type: "number",
        description: "Message priority (1-10, higher is more urgent)",
        defaultValue: 5,
      },
    ],
    handler: async ({ content, priority = 5 }) => {
      await sendMessageToAgent(content, priority);
      return `Message sent to James: "${content}" with priority ${priority}`;
    },
  });

  useCopilotAction({
    name: "get_agent_status",
    description: "Get the current status of the conscious agent James",
    parameters: [],
    handler: async () => {
      const status = await getAgentStatus();
      if (!status) {
        return "Failed to get agent status";
      }
      
      return `James is ${status.is_active ? 'active' : 'inactive'}. ` +
             `Current tasks: ${status.current_tasks?.length || 0}, ` +
             `Queue size: ${status.message_queue_size}, ` +
             `Active subagents: ${status.active_subagents}, ` +
             `Uptime: ${Math.floor(status.uptime_seconds / 60)} minutes`;
    },
  });

  useCopilotAction({
    name: "list_active_tasks",
    description: "List all currently active tasks in the system",
    parameters: [],
    handler: async () => {
      const tasks = await listActiveTasks();
      if (tasks.length === 0) {
        return "No active tasks found";
      }
      
      return `Active tasks (${tasks.length}):\n` +
             tasks.map(task => 
               `- ${task.description} (${task.status}, priority: ${task.priority})`
             ).join('\n');
    },
  });

  useCopilotAction({
    name: "get_subagents",
    description: "Get information about available subagents",
    parameters: [],
    handler: async () => {
      const agents = await getSubagents();
      if (agents.length === 0) {
        return "No subagents found";
      }
      
      return `Available subagents (${agents.length}):\n` +
             agents.map(agent => 
               `- ${agent.name}: ${agent.description} (${agent.is_active ? 'active' : 'inactive'})`
             ).join('\n');
    },
  });

  // Periodically refresh data
  useEffect(() => {
    const interval = setInterval(() => {
      getAgentStatus();
      listActiveTasks();
      getSubagents();
    }, 30000); // Refresh every 30 seconds

    // Initial load
    getAgentStatus();
    listActiveTasks();
    getSubagents();

    return () => clearInterval(interval);
  }, [getAgentStatus, listActiveTasks, getSubagents]);

  return {
    sendMessageToAgent,
    getAgentStatus,
    listActiveTasks,
    getSubagents,
    isLoading,
    error
  };
}