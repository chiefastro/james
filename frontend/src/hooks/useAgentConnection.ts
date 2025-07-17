/**
 * Custom hook for managing WebSocket connection to the conscious agent system.
 * 
 * This hook provides real-time communication with the backend agent system,
 * handling connection state, message sending/receiving, and error recovery.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { wsConfig, errorConfig } from '@/lib/copilot-config';
import { WebSocketMessage, AgentState, ChatMessage } from '@/types';
import { saveMessages, loadMessages, clearMessages as clearStoredMessages } from '@/lib/message-storage';

interface UseAgentConnectionReturn {
  isConnected: boolean;
  agentState: AgentState;
  messages: ChatMessage[];
  sendMessage: (content: string, priority?: number) => Promise<void>;
  reconnect: () => void;
  clearMessages: () => void;
  error: string | null;
}

export function useAgentConnection(): UseAgentConnectionReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [agentState, setAgentState] = useState<AgentState>({
    isConnected: false,
    isTyping: false,
    currentTasks: [],
    queueSize: 0,
  });
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [error, setError] = useState<string | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const heartbeatIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const reconnectAttemptsRef = useRef(0);

  const clearTimeouts = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
      heartbeatIntervalRef.current = null;
    }
  }, []);

  const startHeartbeat = useCallback(() => {
    heartbeatIntervalRef.current = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          type: 'ping',
          timestamp: new Date().toISOString()
        }));
      }
    }, wsConfig.heartbeatInterval);
  }, []);

  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const wsMessage: WebSocketMessage = JSON.parse(event.data);
      
      switch (wsMessage.type) {
        case 'message':
          // Add agent response to messages
          setMessages(prev => [...prev, {
            id: `agent-${Date.now()}`,
            content: wsMessage.data.content || wsMessage.data.message,
            isFromUser: false,
            timestamp: new Date(wsMessage.timestamp),
            status: 'sent'
          }]);
          
          // Update agent typing state
          setAgentState(prev => ({ ...prev, isTyping: false }));
          break;
          
        case 'status':
          // Update agent state
          setAgentState(prev => ({
            ...prev,
            currentTasks: wsMessage.data.current_tasks || [],
            queueSize: wsMessage.data.message_queue_size || 0,
            lastActivity: wsMessage.data.last_activity ? new Date(wsMessage.data.last_activity) : undefined
          }));
          break;
          
        case 'error':
          setError(wsMessage.data.message || 'Unknown error occurred');
          break;
          
        case 'pong':
          // Heartbeat response - connection is alive
          break;
          
        default:
          console.log('Unknown WebSocket message type:', wsMessage.type);
      }
    } catch (err) {
      console.error('Error parsing WebSocket message:', err);
      setError('Failed to parse server message');
    }
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    try {
      wsRef.current = new WebSocket(wsConfig.url);
      
      wsRef.current.onopen = () => {
        setIsConnected(true);
        setAgentState(prev => ({ ...prev, isConnected: true }));
        setError(null);
        reconnectAttemptsRef.current = 0;
        
        // Start heartbeat
        startHeartbeat();
        
        // Subscribe to updates
        wsRef.current?.send(JSON.stringify({
          type: 'subscribe',
          timestamp: new Date().toISOString()
        }));
      };
      
      wsRef.current.onmessage = handleMessage;
      
      wsRef.current.onclose = () => {
        setIsConnected(false);
        setAgentState(prev => ({ ...prev, isConnected: false, isTyping: false }));
        clearTimeouts();
        
        // Attempt reconnection if not at max attempts
        if (reconnectAttemptsRef.current < wsConfig.maxReconnectAttempts) {
          reconnectAttemptsRef.current++;
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, wsConfig.reconnectInterval);
        } else {
          setError('Connection lost. Please refresh the page to reconnect.');
        }
      };
      
      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setError('Connection error occurred');
      };
      
    } catch (err) {
      console.error('Failed to create WebSocket connection:', err);
      setError('Failed to connect to agent system');
    }
  }, [handleMessage, startHeartbeat, clearTimeouts]);

  const sendMessage = useCallback(async (content: string, priority: number = 5): Promise<void> => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      throw new Error('Not connected to agent system');
    }

    // Add user message to chat
    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      content,
      isFromUser: true,
      timestamp: new Date(),
      status: 'sending'
    };
    
    setMessages(prev => [...prev, userMessage]);
    
    try {
      // Send message via WebSocket
      wsRef.current.send(JSON.stringify({
        type: 'message',
        data: {
          content,
          priority,
          source: 'user'
        },
        timestamp: new Date().toISOString()
      }));
      
      // Update message status
      setMessages(prev => prev.map(msg => 
        msg.id === userMessage.id 
          ? { ...msg, status: 'sent' }
          : msg
      ));
      
      // Set agent typing state
      setAgentState(prev => ({ ...prev, isTyping: true }));
      
    } catch (err) {
      console.error('Failed to send message:', err);
      
      // Update message status to error
      setMessages(prev => prev.map(msg => 
        msg.id === userMessage.id 
          ? { ...msg, status: 'error' }
          : msg
      ));
      
      throw new Error('Failed to send message');
    }
  }, []);

  const reconnect = useCallback(() => {
    reconnectAttemptsRef.current = 0;
    setError(null);
    connect();
  }, [connect]);

  const clearMessages = useCallback(() => {
    setMessages([]);
    clearStoredMessages();
  }, []);

  // Load messages from storage on mount
  useEffect(() => {
    const storedMessages = loadMessages();
    if (storedMessages.length > 0) {
      setMessages(storedMessages);
    }
  }, []);

  // Save messages to storage whenever messages change
  useEffect(() => {
    if (messages.length > 0) {
      saveMessages(messages);
    }
  }, [messages]);

  // Initialize connection on mount
  useEffect(() => {
    connect();
    
    return () => {
      clearTimeouts();
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect, clearTimeouts]);

  return {
    isConnected,
    agentState,
    messages,
    sendMessage,
    reconnect,
    clearMessages,
    error
  };
}