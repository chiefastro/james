/**
 * Tests for AgentDashboard component
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { AgentDashboard } from '@/components/AgentDashboard';
import { TaskStatus } from '@/types';
import { useCopilotAgent } from '@/hooks/useCopilotAgent';
import { useAgentConnection } from '@/hooks/useAgentConnection';

// Mock the hooks
jest.mock('@/hooks/useCopilotAgent');
jest.mock('@/hooks/useAgentConnection');

const mockUseCopilotAgent = useCopilotAgent as jest.MockedFunction<typeof useCopilotAgent>;
const mockUseAgentConnection = useAgentConnection as jest.MockedFunction<typeof useAgentConnection>;

describe('AgentDashboard', () => {
  const mockAgentStatus = {
    is_active: true,
    current_tasks: [
      {
        id: 'task-1',
        description: 'Test task 1',
        priority: 1,
        status: TaskStatus.IN_PROGRESS,
        assigned_subagents: ['agent-1'],
        created_at: '2024-01-01T10:00:00Z',
        updated_at: '2024-01-01T10:00:00Z'
      },
      {
        id: 'task-2',
        description: 'Test task 2',
        priority: 2,
        status: TaskStatus.COMPLETED,
        assigned_subagents: ['agent-1', 'agent-2'],
        created_at: '2024-01-01T09:00:00Z',
        updated_at: '2024-01-01T10:30:00Z'
      }
    ],
    message_queue_size: 5,
    active_subagents: 3,
    memory_usage: {
      episodic_count: 10,
      semantic_count: 15,
      procedural_count: 8,
      working_memory_size: 5,
      total_size_mb: 2.5,
      cleanup_last_run: '2024-01-01T08:00:00Z'
    },
    uptime_seconds: 3661,
    last_activity: '2024-01-01T10:30:00Z'
  };

  const mockAgentState = {
    isConnected: true,
    isTyping: false,
    currentTasks: [],
    queueSize: 5,
    lastActivity: new Date()
  };

  beforeEach(() => {
    mockUseCopilotAgent.mockReturnValue({
      getAgentStatus: jest.fn().mockResolvedValue(mockAgentStatus),
      isLoading: false,
      error: null
    });

    mockUseAgentConnection.mockReturnValue({
      isConnected: true,
      agentState: mockAgentState,
      messages: [],
      sendMessage: jest.fn(),
      error: null,
      reconnect: jest.fn()
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('renders dashboard header correctly', async () => {
    render(<AgentDashboard />);
    
    expect(screen.getByText('James Dashboard')).toBeInTheDocument();
    expect(screen.getByText('Connected')).toBeInTheDocument();
    expect(screen.getByText('Refresh')).toBeInTheDocument();
  });

  it('displays status overview cards with correct data', async () => {
    render(<AgentDashboard />);
    
    await waitFor(() => {
      expect(screen.getByText('Active')).toBeInTheDocument();
      expect(screen.getAllByText('5')).toHaveLength(2); // Queue size appears in multiple places
      expect(screen.getByText('2')).toBeInTheDocument(); // Active tasks count
      expect(screen.getByText('3')).toBeInTheDocument(); // Active subagents
    });
  });

  it('displays active tasks with correct information', async () => {
    render(<AgentDashboard />);
    
    await waitFor(() => {
      expect(screen.getAllByText('Active Tasks')).toHaveLength(2); // Appears in card and panel
      expect(screen.getByText('Test task 1')).toBeInTheDocument();
      expect(screen.getByText('Test task 2')).toBeInTheDocument();
      expect(screen.getByText('Priority: 1')).toBeInTheDocument();
      expect(screen.getByText('Priority: 2')).toBeInTheDocument();
    });
  });

  it('displays memory metrics correctly', async () => {
    render(<AgentDashboard />);
    
    await waitFor(() => {
      expect(screen.getAllByText('Memory Usage')).toHaveLength(2); // Appears in system health and memory panel
      expect(screen.getByText('10')).toBeInTheDocument(); // Episodic count
      expect(screen.getByText('15')).toBeInTheDocument(); // Semantic count
      expect(screen.getByText('8')).toBeInTheDocument(); // Procedural count
      expect(screen.getAllByText('5')).toHaveLength(2); // Working memory size and queue size
      expect(screen.getByText('2.5 MB')).toBeInTheDocument(); // Total size
    });
  });

  it('displays system health metrics', async () => {
    render(<AgentDashboard />);
    
    await waitFor(() => {
      expect(screen.getByText('System Health')).toBeInTheDocument();
      expect(screen.getByText('CPU Usage')).toBeInTheDocument();
      expect(screen.getAllByText('Memory Usage')).toHaveLength(2); // Appears in system health and memory panel
      expect(screen.getByText('Network Latency')).toBeInTheDocument();
    });
  });

  it('shows uptime information', async () => {
    render(<AgentDashboard />);
    
    await waitFor(() => {
      expect(screen.getByText('1h 1m 1s')).toBeInTheDocument(); // Formatted uptime
    });
  });

  it('handles refresh button click', async () => {
    const mockGetAgentStatus = jest.fn().mockResolvedValue(mockAgentStatus);
    mockUseCopilotAgent.mockReturnValue({
      getAgentStatus: mockGetAgentStatus,
      isLoading: false,
      error: null
    });

    render(<AgentDashboard />);
    
    // Wait for initial load
    await waitFor(() => {
      expect(mockGetAgentStatus).toHaveBeenCalledTimes(1);
    });
    
    const refreshButton = screen.getByText('Refresh');
    fireEvent.click(refreshButton);
    
    await waitFor(() => {
      expect(mockGetAgentStatus).toHaveBeenCalledTimes(2); // Initial load + refresh
    });
  });

  it('displays error state when data fetch fails', async () => {
    mockUseCopilotAgent.mockReturnValue({
      getAgentStatus: jest.fn().mockRejectedValue(new Error('Fetch failed')),
      isLoading: false,
      error: 'Failed to load agent dashboard data'
    });

    render(<AgentDashboard />);
    
    expect(screen.getByText('Dashboard Error')).toBeInTheDocument();
    expect(screen.getByText('Failed to load agent dashboard data')).toBeInTheDocument();
  });

  it('shows disconnected state when not connected', () => {
    mockUseAgentConnection.mockReturnValue({
      isConnected: false,
      agentState: { ...mockAgentState, isConnected: false },
      messages: [],
      sendMessage: jest.fn(),
      error: null,
      reconnect: jest.fn()
    });

    render(<AgentDashboard />);
    
    expect(screen.getByText('Disconnected')).toBeInTheDocument();
  });

  it('displays typing indicator when agent is typing', () => {
    mockUseAgentConnection.mockReturnValue({
      isConnected: true,
      agentState: { ...mockAgentState, isTyping: true },
      messages: [],
      sendMessage: jest.fn(),
      error: null,
      reconnect: jest.fn()
    });

    render(<AgentDashboard />);
    
    expect(screen.getByText('James is actively processing...')).toBeInTheDocument();
  });

  it('shows loading state', () => {
    mockUseCopilotAgent.mockReturnValue({
      getAgentStatus: jest.fn().mockResolvedValue(mockAgentStatus),
      isLoading: true,
      error: null
    });

    render(<AgentDashboard />);
    
    expect(screen.getByText('Loading dashboard data...')).toBeInTheDocument();
  });

  it('handles empty tasks list', async () => {
    const emptyStatus = { ...mockAgentStatus, current_tasks: [] };
    mockUseCopilotAgent.mockReturnValue({
      getAgentStatus: jest.fn().mockResolvedValue(emptyStatus),
      isLoading: false,
      error: null
    });

    render(<AgentDashboard />);
    
    await waitFor(() => {
      expect(screen.getByText('No active tasks')).toBeInTheDocument();
    });
  });

  it('handles missing memory metrics', async () => {
    const statusWithoutMemory = { ...mockAgentStatus, memory_usage: null };
    mockUseCopilotAgent.mockReturnValue({
      getAgentStatus: jest.fn().mockResolvedValue(statusWithoutMemory),
      isLoading: false,
      error: null
    });

    render(<AgentDashboard />);
    
    await waitFor(() => {
      expect(screen.getByText('Memory metrics unavailable')).toBeInTheDocument();
    });
  });

  it('formats task status icons correctly', async () => {
    render(<AgentDashboard />);
    
    await waitFor(() => {
      // Check that task status icons are rendered (we can't easily test the specific icons)
      const taskElements = screen.getAllByText(/Test task/);
      expect(taskElements).toHaveLength(2);
    });
  });

  it('displays subagent assignments for tasks', async () => {
    render(<AgentDashboard />);
    
    await waitFor(() => {
      expect(screen.getByText('1 subagent(s)')).toBeInTheDocument();
      expect(screen.getByText('2 subagent(s)')).toBeInTheDocument();
    });
  });
});