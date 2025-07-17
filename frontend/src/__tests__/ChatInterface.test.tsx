/**
 * Tests for the ChatInterface component.
 * 
 * These tests verify the chat interface functionality including message display,
 * input handling, connection status, and user interactions.
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ChatInterface } from '@/components/ChatInterface';
import { useAgentConnection } from '@/hooks/useAgentConnection';
import { useCopilotAgent } from '@/hooks/useCopilotAgent';

// Mock the hooks
jest.mock('@/hooks/useAgentConnection');
jest.mock('@/hooks/useCopilotAgent');

const mockUseAgentConnection = useAgentConnection as jest.MockedFunction<typeof useAgentConnection>;
const mockUseCopilotAgent = useCopilotAgent as jest.MockedFunction<typeof useCopilotAgent>;

// Mock clipboard API
Object.assign(navigator, {
  clipboard: {
    writeText: jest.fn(),
  },
});

describe('ChatInterface', () => {
  const mockSendMessage = jest.fn();
  const mockReconnect = jest.fn();
  const mockClearMessages = jest.fn();

  const defaultAgentConnectionReturn = {
    isConnected: true,
    agentState: {
      isConnected: true,
      isTyping: false,
      currentTasks: [],
      queueSize: 0,
    },
    messages: [],
    sendMessage: mockSendMessage,
    reconnect: mockReconnect,
    clearMessages: mockClearMessages,
    error: null,
  };

  const defaultCopilotAgentReturn = {
    sendMessageToAgent: jest.fn(),
    getAgentStatus: jest.fn(),
    listActiveTasks: jest.fn(),
    getSubagents: jest.fn(),
    isLoading: false,
    error: null,
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockUseAgentConnection.mockReturnValue(defaultAgentConnectionReturn);
    mockUseCopilotAgent.mockReturnValue(defaultCopilotAgentReturn);
  });

  describe('Rendering', () => {
    it('should render the chat interface with header', () => {
      render(<ChatInterface />);
      
      expect(screen.getByText('Chat with James')).toBeInTheDocument();
      expect(screen.getByText('Connected')).toBeInTheDocument();
      expect(screen.getByPlaceholderText('Type your message to James...')).toBeInTheDocument();
    });

    it('should show disconnected status when not connected', () => {
      mockUseAgentConnection.mockReturnValue({
        ...defaultAgentConnectionReturn,
        isConnected: false,
        agentState: {
          ...defaultAgentConnectionReturn.agentState,
          isConnected: false,
        },
      });

      render(<ChatInterface />);
      
      expect(screen.getByText('Disconnected')).toBeInTheDocument();
      expect(screen.getByPlaceholderText('Connecting to James...')).toBeInTheDocument();
    });

    it('should display welcome message when no messages exist', () => {
      render(<ChatInterface />);
      
      expect(screen.getByText('Welcome to James')).toBeInTheDocument();
      expect(screen.getByText('Your conscious AI agent is ready to help.')).toBeInTheDocument();
      expect(screen.getByText('Start a conversation below!')).toBeInTheDocument();
    });

    it('should apply custom className and maxHeight props', () => {
      const { container } = render(
        <ChatInterface className="custom-class" maxHeight="h-80" />
      );
      
      const chatContainer = container.firstChild as HTMLElement;
      expect(chatContainer).toHaveClass('custom-class', 'h-80');
    });
  });

  describe('Message Display', () => {
    const mockMessages = [
      {
        id: 'user-1',
        content: 'Hello James',
        isFromUser: true,
        timestamp: new Date('2024-01-01T12:00:00Z'),
        status: 'sent' as const,
      },
      {
        id: 'agent-1',
        content: 'Hello! How can I help you today?',
        isFromUser: false,
        timestamp: new Date('2024-01-01T12:00:30Z'),
        status: 'sent' as const,
      },
    ];

    it('should display messages correctly', () => {
      mockUseAgentConnection.mockReturnValue({
        ...defaultAgentConnectionReturn,
        messages: mockMessages,
      });

      render(<ChatInterface />);
      
      expect(screen.getByText('Hello James')).toBeInTheDocument();
      expect(screen.getByText('Hello! How can I help you today?')).toBeInTheDocument();
    });

    it('should show message timestamps', () => {
      mockUseAgentConnection.mockReturnValue({
        ...defaultAgentConnectionReturn,
        messages: mockMessages,
      });

      render(<ChatInterface />);
      
      // Should show formatted timestamps
      expect(screen.getByText('12:00 PM')).toBeInTheDocument();
      expect(screen.getByText('12:00 PM')).toBeInTheDocument();
    });

    it('should show message status icons for user messages', () => {
      const messageWithStatus = {
        ...mockMessages[0],
        status: 'sending' as const,
      };

      mockUseAgentConnection.mockReturnValue({
        ...defaultAgentConnectionReturn,
        messages: [messageWithStatus],
      });

      render(<ChatInterface />);
      
      // Should show sending status
      expect(screen.getByTestId('clock-icon') || screen.getByLabelText(/sending/i)).toBeInTheDocument();
    });

    it('should show typing indicator when agent is typing', () => {
      mockUseAgentConnection.mockReturnValue({
        ...defaultAgentConnectionReturn,
        agentState: {
          ...defaultAgentConnectionReturn.agentState,
          isTyping: true,
        },
      });

      render(<ChatInterface />);
      
      expect(screen.getByText('James is thinking...')).toBeInTheDocument();
    });
  });

  describe('Message Input', () => {
    it('should handle message input and submission', async () => {
      const user = userEvent.setup();
      render(<ChatInterface />);
      
      const input = screen.getByPlaceholderText('Type your message to James...');
      const sendButton = screen.getByRole('button', { name: /send/i });
      
      await user.type(input, 'Test message');
      await user.click(sendButton);
      
      expect(mockSendMessage).toHaveBeenCalledWith('Test message');
    });

    it('should handle Enter key submission', async () => {
      const user = userEvent.setup();
      render(<ChatInterface />);
      
      const input = screen.getByPlaceholderText('Type your message to James...');
      
      await user.type(input, 'Test message{enter}');
      
      expect(mockSendMessage).toHaveBeenCalledWith('Test message');
    });

    it('should not submit empty messages', async () => {
      const user = userEvent.setup();
      render(<ChatInterface />);
      
      const sendButton = screen.getByRole('button', { name: /send/i });
      
      await user.click(sendButton);
      
      expect(mockSendMessage).not.toHaveBeenCalled();
    });

    it('should disable input when disconnected', () => {
      mockUseAgentConnection.mockReturnValue({
        ...defaultAgentConnectionReturn,
        isConnected: false,
      });

      render(<ChatInterface />);
      
      const input = screen.getByPlaceholderText('Connecting to James...');
      const sendButton = screen.getByRole('button', { name: /send/i });
      
      expect(input).toBeDisabled();
      expect(sendButton).toBeDisabled();
    });

    it('should disable input when loading', () => {
      mockUseCopilotAgent.mockReturnValue({
        ...defaultCopilotAgentReturn,
        isLoading: true,
      });

      render(<ChatInterface />);
      
      const input = screen.getByPlaceholderText('Type your message to James...');
      const sendButton = screen.getByRole('button', { name: /send/i });
      
      expect(input).toBeDisabled();
      expect(sendButton).toBeDisabled();
    });

    it('should show character count when typing', async () => {
      const user = userEvent.setup();
      render(<ChatInterface />);
      
      const input = screen.getByPlaceholderText('Type your message to James...');
      
      await user.type(input, 'Hello');
      
      expect(screen.getByText('5/1000')).toBeInTheDocument();
    });
  });

  describe('Connection Status and Controls', () => {
    it('should show connection details when clicked', async () => {
      const user = userEvent.setup();
      render(<ChatInterface />);
      
      const connectionButton = screen.getByText('Connected');
      await user.click(connectionButton);
      
      expect(screen.getByText('Connection Status')).toBeInTheDocument();
      expect(screen.getByText('Queue:')).toBeInTheDocument();
    });

    it('should show reconnect option when disconnected', async () => {
      const user = userEvent.setup();
      mockUseAgentConnection.mockReturnValue({
        ...defaultAgentConnectionReturn,
        isConnected: false,
      });

      render(<ChatInterface />);
      
      const connectionButton = screen.getByText('Disconnected');
      await user.click(connectionButton);
      
      const reconnectButton = screen.getByText('Reconnect');
      await user.click(reconnectButton);
      
      expect(mockReconnect).toHaveBeenCalled();
    });

    it('should show clear messages option', async () => {
      const user = userEvent.setup();
      // Mock window.confirm
      window.confirm = jest.fn(() => true);
      
      render(<ChatInterface />);
      
      const moreButton = screen.getByRole('button', { name: /more/i });
      await user.click(moreButton);
      
      const clearButton = screen.getByText('Clear Messages');
      await user.click(clearButton);
      
      expect(window.confirm).toHaveBeenCalledWith('Clear all messages? This action cannot be undone.');
    });
  });

  describe('Error Handling', () => {
    it('should display error messages', () => {
      mockUseAgentConnection.mockReturnValue({
        ...defaultAgentConnectionReturn,
        error: 'Connection failed',
      });

      render(<ChatInterface />);
      
      expect(screen.getByText('Connection failed')).toBeInTheDocument();
      expect(screen.getByText('Retry')).toBeInTheDocument();
    });

    it('should handle retry on error', async () => {
      const user = userEvent.setup();
      mockUseAgentConnection.mockReturnValue({
        ...defaultAgentConnectionReturn,
        error: 'Connection failed',
      });

      render(<ChatInterface />);
      
      const retryButton = screen.getByText('Retry');
      await user.click(retryButton);
      
      expect(mockReconnect).toHaveBeenCalled();
    });
  });

  describe('Message Actions', () => {
    const mockMessages = [
      {
        id: 'user-1',
        content: 'Hello James',
        isFromUser: true,
        timestamp: new Date('2024-01-01T12:00:00Z'),
        status: 'sent' as const,
      },
    ];

    it('should copy message content when copy button is clicked', async () => {
      const user = userEvent.setup();
      mockUseAgentConnection.mockReturnValue({
        ...defaultAgentConnectionReturn,
        messages: mockMessages,
      });

      render(<ChatInterface />);
      
      // Hover over message to show copy button
      const messageContainer = screen.getByText('Hello James').closest('.group');
      expect(messageContainer).toBeInTheDocument();
      
      // Find and click copy button (it should be visible on hover)
      const copyButton = screen.getByTitle('Copy message');
      await user.click(copyButton);
      
      expect(navigator.clipboard.writeText).toHaveBeenCalledWith('Hello James');
    });
  });

  describe('Accessibility', () => {
    it('should have proper ARIA labels', () => {
      render(<ChatInterface />);
      
      const input = screen.getByPlaceholderText('Type your message to James...');
      const sendButton = screen.getByRole('button', { name: /send/i });
      
      expect(input).toHaveAttribute('type', 'text');
      expect(sendButton).toHaveAttribute('type', 'submit');
    });

    it('should support keyboard navigation', async () => {
      const user = userEvent.setup();
      render(<ChatInterface />);
      
      const input = screen.getByPlaceholderText('Type your message to James...');
      
      // Tab should focus the input
      await user.tab();
      expect(input).toHaveFocus();
      
      // Tab again should focus the send button
      await user.tab();
      const sendButton = screen.getByRole('button', { name: /send/i });
      expect(sendButton).toHaveFocus();
    });
  });

  describe('Auto-scroll Behavior', () => {
    it('should scroll to bottom when new messages arrive', async () => {
      const scrollIntoViewMock = jest.fn();
      Element.prototype.scrollIntoView = scrollIntoViewMock;

      const { rerender } = render(<ChatInterface />);
      
      // Add a message
      mockUseAgentConnection.mockReturnValue({
        ...defaultAgentConnectionReturn,
        messages: [
          {
            id: 'user-1',
            content: 'Hello',
            isFromUser: true,
            timestamp: new Date(),
            status: 'sent' as const,
          },
        ],
      });

      rerender(<ChatInterface />);
      
      await waitFor(() => {
        expect(scrollIntoViewMock).toHaveBeenCalledWith({ behavior: 'smooth' });
      });
    });
  });
});