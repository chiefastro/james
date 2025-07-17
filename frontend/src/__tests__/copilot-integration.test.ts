/**
 * Integration tests for CopilotKit functionality.
 * 
 * These tests verify the CopilotKit integration with the conscious agent system,
 * including function calling, error handling, and real-time communication.
 */

import { renderHook, act } from '@testing-library/react';
import { useCopilotAgent } from '@/hooks/useCopilotAgent';
import { useAgentConnection } from '@/hooks/useAgentConnection';

// Mock fetch for API calls
global.fetch = jest.fn();

// Mock WebSocket
class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  readyState = MockWebSocket.CONNECTING;
  onopen: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;

  constructor(public url: string) {
    setTimeout(() => {
      this.readyState = MockWebSocket.OPEN;
      if (this.onopen) {
        this.onopen(new Event('open'));
      }
    }, 100);
  }

  send(data: string) {
    // Mock sending data
    console.log('Mock WebSocket send:', data);
  }

  close() {
    this.readyState = MockWebSocket.CLOSED;
    if (this.onclose) {
      this.onclose(new CloseEvent('close'));
    }
  }
}

global.WebSocket = MockWebSocket as any;

describe('CopilotKit Integration', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (fetch as jest.Mock).mockClear();
  });

  describe('useCopilotAgent Hook', () => {
    it('should send message to agent successfully', async () => {
      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          status: 'queued',
          message_id: 'test-message-id'
        })
      });

      const { result } = renderHook(() => useCopilotAgent());

      await act(async () => {
        await result.current.sendMessageToAgent('Hello James', 5);
      });

      expect(fetch).toHaveBeenCalledWith(
        'http://localhost:8000/message',
        expect.objectContaining({
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            content: 'Hello James',
            priority: 5,
            source: 'USER',
            metadata: expect.objectContaining({
              frontend_source: 'copilot'
            })
          })
        })
      );
    });

    it('should handle API errors gracefully', async () => {
      (fetch as jest.Mock).mockRejectedValueOnce(new Error('Network error'));

      const { result } = renderHook(() => useCopilotAgent());

      await act(async () => {
        try {
          await result.current.sendMessageToAgent('Hello James');
        } catch (error) {
          expect(error).toBeInstanceOf(Error);
          expect((error as Error).message).toBe('Network error');
        }
      });

      expect(result.current.error).toBe('Network error');
    });

    it('should get agent status successfully', async () => {
      const mockStatus = {
        is_active: true,
        current_tasks: [],
        message_queue_size: 2,
        active_subagents: 3,
        memory_usage: {},
        uptime_seconds: 3600,
        last_activity: new Date().toISOString()
      };

      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockStatus
      });

      const { result } = renderHook(() => useCopilotAgent());

      let status;
      await act(async () => {
        status = await result.current.getAgentStatus();
      });

      expect(status).toEqual(mockStatus);
      expect(fetch).toHaveBeenCalledWith(
        'http://localhost:8000/agent/status',
        expect.objectContaining({
          headers: {
            'Content-Type': 'application/json'
          }
        })
      );
    });

    it('should list active tasks successfully', async () => {
      const mockTasks = [
        {
          id: 'task-1',
          description: 'Test task',
          status: 'IN_PROGRESS',
          priority: 5,
          assigned_subagents: ['agent-1'],
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        }
      ];

      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockTasks
      });

      const { result } = renderHook(() => useCopilotAgent());

      let tasks;
      await act(async () => {
        tasks = await result.current.listActiveTasks();
      });

      expect(tasks).toEqual(mockTasks);
    });

    it('should get subagents successfully', async () => {
      const mockSubagents = [
        {
          id: 'subagent-1',
          name: 'Reflection Agent',
          description: 'Agent for self-reflection and analysis',
          capabilities: ['reflection', 'analysis'],
          is_active: true,
          created_at: new Date().toISOString(),
          input_schema: {},
          output_schema: {},
          import_path: 'backend.agents.reflection'
        }
      ];

      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockSubagents
      });

      const { result } = renderHook(() => useCopilotAgent());

      let subagents;
      await act(async () => {
        subagents = await result.current.getSubagents();
      });

      expect(subagents).toEqual(mockSubagents);
    });
  });

  describe('useAgentConnection Hook', () => {
    it('should establish WebSocket connection', async () => {
      const { result } = renderHook(() => useAgentConnection());

      // Wait for connection to establish
      await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 150));
      });

      expect(result.current.isConnected).toBe(true);
      expect(result.current.agentState.isConnected).toBe(true);
    });

    it('should send message via WebSocket', async () => {
      const { result } = renderHook(() => useAgentConnection());

      // Wait for connection
      await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 150));
      });

      await act(async () => {
        await result.current.sendMessage('Test message', 5);
      });

      expect(result.current.messages).toHaveLength(1);
      expect(result.current.messages[0]).toMatchObject({
        content: 'Test message',
        isFromUser: true,
        status: 'sent'
      });
    });

    it('should handle WebSocket message reception', async () => {
      const { result } = renderHook(() => useAgentConnection());

      // Wait for connection
      await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 150));
      });

      // Simulate receiving a message
      const mockMessage = {
        type: 'message',
        data: {
          content: 'Hello from James',
          source: 'agent'
        },
        timestamp: new Date().toISOString()
      };

      // Access the WebSocket instance and simulate message
      const wsInstance = (global.WebSocket as any).mock.instances[0];
      
      await act(async () => {
        if (wsInstance.onmessage) {
          wsInstance.onmessage({
            data: JSON.stringify(mockMessage)
          } as MessageEvent);
        }
      });

      expect(result.current.messages).toContainEqual(
        expect.objectContaining({
          content: 'Hello from James',
          isFromUser: false
        })
      );
    });

    it('should handle connection errors', async () => {
      const { result } = renderHook(() => useAgentConnection());

      // Simulate connection error
      await act(async () => {
        const wsInstance = (global.WebSocket as any).mock.instances[0];
        if (wsInstance.onerror) {
          wsInstance.onerror(new Event('error'));
        }
      });

      expect(result.current.error).toBeTruthy();
    });

    it('should attempt reconnection on disconnect', async () => {
      const { result } = renderHook(() => useAgentConnection());

      // Wait for initial connection
      await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 150));
      });

      expect(result.current.isConnected).toBe(true);

      // Simulate disconnect
      await act(async () => {
        const wsInstance = (global.WebSocket as any).mock.instances[0];
        if (wsInstance.onclose) {
          wsInstance.onclose(new CloseEvent('close'));
        }
      });

      expect(result.current.isConnected).toBe(false);
      expect(result.current.agentState.isConnected).toBe(false);
    });
  });

  describe('Error Handling', () => {
    it('should handle network timeouts', async () => {
      (fetch as jest.Mock).mockImplementationOnce(() => 
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Timeout')), 100)
        )
      );

      const { result } = renderHook(() => useCopilotAgent());

      await act(async () => {
        try {
          await result.current.sendMessageToAgent('Test message');
        } catch (error) {
          expect((error as Error).message).toBe('Timeout');
        }
      });
    });

    it('should handle malformed API responses', async () => {
      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error'
      });

      const { result } = renderHook(() => useCopilotAgent());

      await act(async () => {
        try {
          await result.current.getAgentStatus();
        } catch (error) {
          expect((error as Error).message).toContain('500');
        }
      });
    });
  });

  describe('Real-time Updates', () => {
    it('should update agent state from WebSocket messages', async () => {
      const { result } = renderHook(() => useAgentConnection());

      // Wait for connection
      await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 150));
      });

      // Simulate status update
      const statusUpdate = {
        type: 'status',
        data: {
          current_tasks: [{ id: 'task-1', description: 'New task' }],
          message_queue_size: 5,
          last_activity: new Date().toISOString()
        },
        timestamp: new Date().toISOString()
      };

      await act(async () => {
        const wsInstance = (global.WebSocket as any).mock.instances[0];
        if (wsInstance.onmessage) {
          wsInstance.onmessage({
            data: JSON.stringify(statusUpdate)
          } as MessageEvent);
        }
      });

      expect(result.current.agentState.currentTasks).toHaveLength(1);
      expect(result.current.agentState.queueSize).toBe(5);
    });
  });
});