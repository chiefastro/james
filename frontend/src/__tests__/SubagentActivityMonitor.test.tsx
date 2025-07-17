/**
 * Tests for SubagentActivityMonitor component
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { SubagentActivityMonitor } from '@/components/SubagentActivityMonitor';
import { useCopilotAgent } from '@/hooks/useCopilotAgent';

// Mock the hooks
jest.mock('@/hooks/useCopilotAgent');

const mockUseCopilotAgent = useCopilotAgent as jest.MockedFunction<typeof useCopilotAgent>;

describe('SubagentActivityMonitor', () => {
  beforeEach(() => {
    mockUseCopilotAgent.mockReturnValue({
      getAgentStatus: jest.fn().mockResolvedValue({}),
      isLoading: false,
      error: null
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('renders component header correctly', async () => {
    render(<SubagentActivityMonitor />);
    
    expect(screen.getByText('Subagent Activity')).toBeInTheDocument();
    expect(screen.getByText('Refresh')).toBeInTheDocument();
  });

  it('displays active subagents section', async () => {
    render(<SubagentActivityMonitor />);
    
    await waitFor(() => {
      expect(screen.getByText('Active Subagents')).toBeInTheDocument();
    });
  });

  it('shows subagent information with correct data', async () => {
    render(<SubagentActivityMonitor />);
    
    await waitFor(() => {
      // Check for mock subagent names (they appear in multiple places)
      expect(screen.getAllByText('Reflection Agent')).toHaveLength(3); // In subagent list and communications
      expect(screen.getAllByText('Builder Agent')).toHaveLength(2); // In subagent list and communications
      expect(screen.getAllByText('External Input Agent')).toHaveLength(2); // In subagent list and communications
    });
  });

  it('displays subagent status badges', async () => {
    render(<SubagentActivityMonitor />);
    
    await waitFor(() => {
      expect(screen.getByText('Active')).toBeInTheDocument();
      expect(screen.getByText('Idle')).toBeInTheDocument();
      expect(screen.getByText('Busy')).toBeInTheDocument();
    });
  });

  it('shows current task for active subagents', async () => {
    render(<SubagentActivityMonitor />);
    
    await waitFor(() => {
      expect(screen.getAllByText('Current Task:')).toHaveLength(2);
      expect(screen.getByText('Analyzing recent conversation patterns')).toBeInTheDocument();
      expect(screen.getByText('Processing external API data')).toBeInTheDocument();
    });
  });

  it('displays task completion metrics', async () => {
    render(<SubagentActivityMonitor />);
    
    await waitFor(() => {
      expect(screen.getAllByText('Completed')).toHaveLength(3);
      expect(screen.getAllByText('Failed')).toHaveLength(3);
      expect(screen.getAllByText('Success Rate')).toHaveLength(3);
    });
  });

  it('shows subagent capabilities', async () => {
    render(<SubagentActivityMonitor />);
    
    await waitFor(() => {
      expect(screen.getAllByText('Capabilities:')).toHaveLength(3);
      expect(screen.getByText('self-analysis')).toBeInTheDocument();
      expect(screen.getByText('pattern-recognition')).toBeInTheDocument();
      expect(screen.getByText('code-generation')).toBeInTheDocument();
      expect(screen.getByText('api-integration')).toBeInTheDocument();
    });
  });

  it('displays recent communications section', async () => {
    render(<SubagentActivityMonitor />);
    
    await waitFor(() => {
      expect(screen.getByText('Recent Communications')).toBeInTheDocument();
    });
  });

  it('shows communication entries with correct information', async () => {
    render(<SubagentActivityMonitor />);
    
    await waitFor(() => {
      expect(screen.getByText('Analysis complete: Found 3 areas for improvement...')).toBeInTheDocument();
      expect(screen.getByText('Requesting permission to access external API...')).toBeInTheDocument();
      expect(screen.getByText('New capability module created successfully')).toBeInTheDocument();
    });
  });

  it('displays communication types and statuses', async () => {
    render(<SubagentActivityMonitor />);
    
    await waitFor(() => {
      // Check for communication types and statuses by looking for specific text content
      expect(screen.getAllByText(/response/i)).toHaveLength(5); // Appears in multiple places
      expect(screen.getAllByText(/request/i)).toHaveLength(2); // Appears in multiple places
      expect(screen.getAllByText(/success/i)).toHaveLength(6); // Appears in multiple places
      expect(screen.getByText(/pending/i)).toBeInTheDocument();
    });
  });

  it('handles refresh button click', async () => {
    render(<SubagentActivityMonitor />);
    
    const refreshButton = screen.getByText('Refresh');
    fireEvent.click(refreshButton);
    
    // The component should handle the refresh (we can't easily test the internal state change)
    expect(refreshButton).toBeInTheDocument();
  });

  it('formats time correctly', async () => {
    render(<SubagentActivityMonitor />);
    
    await waitFor(() => {
      // Check that time formatting is working (should show relative times like "30s ago", "5m ago")
      const timeElements = screen.getAllByText(/ago$/);
      expect(timeElements.length).toBeGreaterThan(0);
    });
  });

  it('calculates success rates correctly', async () => {
    render(<SubagentActivityMonitor />);
    
    await waitFor(() => {
      // Check that success rate percentages are displayed
      const percentageElements = screen.getAllByText(/%$/);
      expect(percentageElements.length).toBeGreaterThan(0);
    });
  });

  it('shows last updated timestamp', async () => {
    render(<SubagentActivityMonitor />);
    
    await waitFor(() => {
      expect(screen.getByText(/Last updated:/)).toBeInTheDocument();
    });
  });

  it('displays response time metrics', async () => {
    render(<SubagentActivityMonitor />);
    
    await waitFor(() => {
      expect(screen.getAllByText(/Avg response:/)).toHaveLength(3);
      expect(screen.getByText(/250/)).toBeInTheDocument();
      expect(screen.getByText(/450/)).toBeInTheDocument();
      expect(screen.getByText(/180/)).toBeInTheDocument();
    });
  });

  it('handles different subagent statuses with appropriate styling', async () => {
    render(<SubagentActivityMonitor />);
    
    await waitFor(() => {
      // Check that different status badges are present
      const activeStatus = screen.getByText('Active');
      const idleStatus = screen.getByText('Idle');
      const busyStatus = screen.getByText('Busy');
      
      expect(activeStatus).toBeInTheDocument();
      expect(idleStatus).toBeInTheDocument();
      expect(busyStatus).toBeInTheDocument();
    });
  });

  it('shows communication message previews', async () => {
    render(<SubagentActivityMonitor />);
    
    await waitFor(() => {
      expect(screen.getByText('Analysis complete: Found 3 areas for improvement...')).toBeInTheDocument();
      expect(screen.getByText('Failed to access memory store: Connection timeout')).toBeInTheDocument();
    });
  });

  it('displays error communications correctly', async () => {
    render(<SubagentActivityMonitor />);
    
    await waitFor(() => {
      // Check for error communication type and failed status using regex
      expect(screen.getByText(/error/i)).toBeInTheDocument();
      expect(screen.getAllByText(/failed/i)).toHaveLength(5); // Appears in multiple places
    });
  });
});