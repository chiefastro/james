/**
 * Enhanced chat interface component for the Conscious Agent System.
 * 
 * This component provides a comprehensive chat experience with real-time messaging,
 * message history, typing indicators, and connection status display.
 */

"use client";

import { useState, useRef, useEffect } from 'react';
import { useAgentConnection } from '@/hooks/useAgentConnection';
import { useCopilotAgent } from '@/hooks/useCopilotAgent';
import { 
  MessageSquare, 
  Send, 
  Wifi, 
  WifiOff, 
  Clock, 
  CheckCircle, 
  XCircle,
  MoreVertical,
  Trash2,
  Copy
} from 'lucide-react';
import { ChatMessage } from '@/types';

interface ChatInterfaceProps {
  className?: string;
  maxHeight?: string;
}

export function ChatInterface({ className = '', maxHeight = 'h-96' }: ChatInterfaceProps) {
  const { messages, sendMessage, isConnected, agentState, error, reconnect } = useAgentConnection();
  const { isLoading } = useCopilotAgent();
  const [inputMessage, setInputMessage] = useState('');
  const [showConnectionDetails, setShowConnectionDetails] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus input when connected
  useEffect(() => {
    if (isConnected && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isConnected]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMessage.trim() || !isConnected || isLoading) return;

    try {
      await sendMessage(inputMessage.trim());
      setInputMessage('');
    } catch (error) {
      console.error('Failed to send message:', error);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(e);
    }
  };

  const copyMessage = (content: string) => {
    navigator.clipboard.writeText(content);
  };

  const clearMessages = () => {
    // This would typically call a function to clear message history
    // For now, we'll just show a confirmation
    if (window.confirm('Clear all messages? This action cannot be undone.')) {
      // Implementation would go here
      console.log('Clear messages requested');
    }
  };

  const formatTimestamp = (timestamp: Date) => {
    const now = new Date();
    const diff = now.getTime() - timestamp.getTime();
    
    if (diff < 60000) { // Less than 1 minute
      return 'Just now';
    } else if (diff < 3600000) { // Less than 1 hour
      return `${Math.floor(diff / 60000)}m ago`;
    } else if (diff < 86400000) { // Less than 1 day
      return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } else {
      return timestamp.toLocaleDateString();
    }
  };

  const getMessageStatusIcon = (message: ChatMessage) => {
    if (!message.isFromUser) return null;
    
    switch (message.status) {
      case 'sending':
        return <Clock className="h-3 w-3 text-gray-400 animate-pulse" />;
      case 'sent':
        return <CheckCircle className="h-3 w-3 text-green-500" />;
      case 'error':
        return <XCircle className="h-3 w-3 text-red-500" />;
      default:
        return null;
    }
  };

  return (
    <div className={`bg-white rounded-lg shadow-md flex flex-col ${maxHeight} ${className}`}>
      {/* Chat Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200">
        <div className="flex items-center space-x-2">
          <MessageSquare className="h-5 w-5 text-blue-600" />
          <h3 className="font-semibold text-gray-900">Chat with James</h3>
        </div>
        
        <div className="flex items-center space-x-2">
          {/* Connection Status */}
          <button
            onClick={() => setShowConnectionDetails(!showConnectionDetails)}
            className={`flex items-center space-x-2 px-2 py-1 rounded-md text-sm transition-colors ${
              isConnected 
                ? 'text-green-600 hover:bg-green-50' 
                : 'text-red-600 hover:bg-red-50'
            }`}
          >
            {isConnected ? (
              <Wifi className="h-4 w-4" />
            ) : (
              <WifiOff className="h-4 w-4" />
            )}
            <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
          </button>

          {/* Menu */}
          <div className="relative">
            <button
              onClick={() => setShowConnectionDetails(!showConnectionDetails)}
              className="p-1 rounded-md hover:bg-gray-100"
            >
              <MoreVertical className="h-4 w-4 text-gray-500" />
            </button>
            
            {showConnectionDetails && (
              <div className="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg border border-gray-200 z-10">
                <div className="py-1">
                  <div className="px-3 py-2 text-xs text-gray-500 border-b">
                    Connection Status
                  </div>
                  <div className="px-3 py-2 text-sm">
                    <div className="flex justify-between">
                      <span>Status:</span>
                      <span className={isConnected ? 'text-green-600' : 'text-red-600'}>
                        {isConnected ? 'Connected' : 'Disconnected'}
                      </span>
                    </div>
                    <div className="flex justify-between mt-1">
                      <span>Queue:</span>
                      <span>{agentState.queueSize}</span>
                    </div>
                  </div>
                  {!isConnected && (
                    <button
                      onClick={reconnect}
                      className="w-full px-3 py-2 text-left text-sm hover:bg-gray-50 text-blue-600"
                    >
                      Reconnect
                    </button>
                  )}
                  <button
                    onClick={clearMessages}
                    className="w-full px-3 py-2 text-left text-sm hover:bg-gray-50 text-red-600 flex items-center"
                  >
                    <Trash2 className="h-4 w-4 mr-2" />
                    Clear Messages
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="px-4 py-2 bg-red-50 border-b border-red-200">
          <div className="flex items-center justify-between">
            <span className="text-red-700 text-sm">{error}</span>
            <button
              onClick={reconnect}
              className="text-red-600 hover:text-red-800 text-sm underline"
            >
              Retry
            </button>
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="text-center text-gray-500 py-8">
            <MessageSquare className="h-12 w-12 mx-auto mb-4 text-gray-300" />
            <p className="text-lg font-medium mb-2">Welcome to James</p>
            <p className="text-sm">Your conscious AI agent is ready to help.</p>
            <p className="text-sm text-gray-400 mt-2">Start a conversation below!</p>
          </div>
        ) : (
          <>
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.isFromUser ? 'justify-end' : 'justify-start'}`}
              >
                <div className="group relative max-w-xs lg:max-w-md">
                  <div
                    className={`px-4 py-2 rounded-lg ${
                      message.isFromUser
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-200 text-gray-900'
                    }`}
                  >
                    <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                    <div className={`flex items-center justify-between mt-1 text-xs ${
                      message.isFromUser ? 'text-blue-100' : 'text-gray-500'
                    }`}>
                      <span>{formatTimestamp(message.timestamp)}</span>
                      <div className="flex items-center space-x-1">
                        {getMessageStatusIcon(message)}
                      </div>
                    </div>
                  </div>
                  
                  {/* Message Actions */}
                  <div className="absolute top-0 right-0 -mr-8 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={() => copyMessage(message.content)}
                      className="p-1 rounded-md hover:bg-gray-100"
                      title="Copy message"
                    >
                      <Copy className="h-3 w-3 text-gray-500" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
            
            {/* Typing Indicator */}
            {agentState.isTyping && (
              <div className="flex justify-start">
                <div className="bg-gray-200 text-gray-900 px-4 py-2 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" />
                      <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                      <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                    </div>
                    <span className="text-sm text-gray-600">James is thinking...</span>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Message Input */}
      <form onSubmit={handleSendMessage} className="p-4 border-t border-gray-200">
        <div className="flex space-x-2">
          <input
            ref={inputRef}
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              isConnected 
                ? "Type your message to James..." 
                : "Connecting to James..."
            }
            disabled={!isConnected || isLoading}
            className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
          />
          <button
            type="submit"
            disabled={!isConnected || !inputMessage.trim() || isLoading}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center"
          >
            {isLoading ? (
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </button>
        </div>
        
        {/* Input Helper Text */}
        <div className="flex justify-between items-center mt-2 text-xs text-gray-500">
          <span>Press Enter to send, Shift+Enter for new line</span>
          {inputMessage.length > 0 && (
            <span>{inputMessage.length}/1000</span>
          )}
        </div>
      </form>
    </div>
  );
}