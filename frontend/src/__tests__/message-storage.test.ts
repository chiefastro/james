/**
 * Tests for message storage utilities.
 * 
 * These tests verify the localStorage-based message persistence functionality
 * including saving, loading, clearing, and data validation.
 */

import {
  saveMessages,
  loadMessages,
  clearMessages,
  getStorageInfo,
  exportMessages,
  importMessages,
  cleanupOldMessages,
} from '@/lib/message-storage';
import { ChatMessage } from '@/types';

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};

  return {
    getItem: jest.fn((key: string) => store[key] || null),
    setItem: jest.fn((key: string, value: string) => {
      store[key] = value;
    }),
    removeItem: jest.fn((key: string) => {
      delete store[key];
    }),
    clear: jest.fn(() => {
      store = {};
    }),
  };
})();

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
});

// Mock URL and Blob for export functionality
global.URL = {
  createObjectURL: jest.fn(() => 'mock-url'),
  revokeObjectURL: jest.fn(),
} as any;

global.Blob = jest.fn((content, options) => ({
  content,
  options,
  size: JSON.stringify(content).length,
})) as any;

describe('Message Storage', () => {
  const mockMessages: ChatMessage[] = [
    {
      id: 'user-1',
      content: 'Hello James',
      isFromUser: true,
      timestamp: new Date('2024-01-01T12:00:00Z'),
      status: 'sent',
    },
    {
      id: 'agent-1',
      content: 'Hello! How can I help you?',
      isFromUser: false,
      timestamp: new Date('2024-01-01T12:00:30Z'),
      status: 'sent',
    },
  ];

  beforeEach(() => {
    jest.clearAllMocks();
    localStorageMock.clear();
  });

  describe('saveMessages', () => {
    it('should save messages to localStorage', () => {
      saveMessages(mockMessages);

      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        'james-chat-messages',
        expect.stringContaining('Hello James')
      );
    });

    it('should handle localStorage errors gracefully', () => {
      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();
      localStorageMock.setItem.mockImplementationOnce(() => {
        throw new Error('Storage quota exceeded');
      });

      expect(() => saveMessages(mockMessages)).not.toThrow();
      expect(consoleSpy).toHaveBeenCalledWith(
        'Failed to save messages to localStorage:',
        expect.any(Error)
      );

      consoleSpy.mockRestore();
    });

    it('should limit stored messages to maximum count', () => {
      const manyMessages = Array.from({ length: 1500 }, (_, i) => ({
        id: `msg-${i}`,
        content: `Message ${i}`,
        isFromUser: i % 2 === 0,
        timestamp: new Date(),
        status: 'sent' as const,
      }));

      saveMessages(manyMessages);

      const savedData = JSON.parse(localStorageMock.setItem.mock.calls[0][1]);
      expect(savedData).toHaveLength(1000); // MAX_STORED_MESSAGES
    });

    it('should convert timestamps to ISO strings', () => {
      saveMessages(mockMessages);

      const savedData = JSON.parse(localStorageMock.setItem.mock.calls[0][1]);
      expect(savedData[0].timestamp).toBe('2024-01-01T12:00:00.000Z');
    });
  });

  describe('loadMessages', () => {
    it('should load messages from localStorage', () => {
      const storedData = [
        {
          id: 'user-1',
          content: 'Hello James',
          isFromUser: true,
          timestamp: '2024-01-01T12:00:00.000Z',
          status: 'sent',
        },
      ];

      localStorageMock.getItem.mockReturnValueOnce(JSON.stringify(storedData));

      const messages = loadMessages();

      expect(messages).toHaveLength(1);
      expect(messages[0].content).toBe('Hello James');
      expect(messages[0].timestamp).toBeInstanceOf(Date);
    });

    it('should return empty array when no data exists', () => {
      localStorageMock.getItem.mockReturnValueOnce(null);

      const messages = loadMessages();

      expect(messages).toEqual([]);
    });

    it('should handle corrupted data gracefully', () => {
      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();
      localStorageMock.getItem.mockReturnValueOnce('invalid json');

      const messages = loadMessages();

      expect(messages).toEqual([]);
      expect(consoleSpy).toHaveBeenCalledWith(
        'Failed to load messages from localStorage:',
        expect.any(Error)
      );

      consoleSpy.mockRestore();
    });

    it('should filter out invalid messages', () => {
      const mixedData = [
        {
          id: 'valid-1',
          content: 'Valid message',
          isFromUser: true,
          timestamp: '2024-01-01T12:00:00.000Z',
        },
        {
          // Missing required fields
          content: 'Invalid message',
        },
        {
          id: 'valid-2',
          content: 'Another valid message',
          isFromUser: false,
          timestamp: '2024-01-01T12:01:00.000Z',
        },
      ];

      localStorageMock.getItem.mockReturnValueOnce(JSON.stringify(mixedData));

      const messages = loadMessages();

      expect(messages).toHaveLength(2);
      expect(messages[0].content).toBe('Valid message');
      expect(messages[1].content).toBe('Another valid message');
    });
  });

  describe('clearMessages', () => {
    it('should remove messages from localStorage', () => {
      clearMessages();

      expect(localStorageMock.removeItem).toHaveBeenCalledWith('james-chat-messages');
    });

    it('should handle localStorage errors gracefully', () => {
      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();
      localStorageMock.removeItem.mockImplementationOnce(() => {
        throw new Error('Storage error');
      });

      expect(() => clearMessages()).not.toThrow();
      expect(consoleSpy).toHaveBeenCalledWith(
        'Failed to clear messages from localStorage:',
        expect.any(Error)
      );

      consoleSpy.mockRestore();
    });
  });

  describe('getStorageInfo', () => {
    it('should return storage information', () => {
      const storedData = [
        {
          id: 'user-1',
          content: 'Hello',
          isFromUser: true,
          timestamp: '2024-01-01T12:00:00.000Z',
        },
      ];

      localStorageMock.getItem.mockReturnValueOnce(JSON.stringify(storedData));

      const info = getStorageInfo();

      expect(info.messageCount).toBe(1);
      expect(info.storageSize).toBeGreaterThan(0);
    });

    it('should return zero values when no data exists', () => {
      localStorageMock.getItem.mockReturnValueOnce(null);

      const info = getStorageInfo();

      expect(info.messageCount).toBe(0);
      expect(info.storageSize).toBe(0);
    });
  });

  describe('exportMessages', () => {
    // Mock DOM elements for download
    const mockLink = {
      href: '',
      download: '',
      click: jest.fn(),
    };

    beforeEach(() => {
      document.createElement = jest.fn().mockReturnValue(mockLink);
      document.body.appendChild = jest.fn();
      document.body.removeChild = jest.fn();
    });

    it('should create download link for messages export', () => {
      const storedData = [mockMessages[0]];
      localStorageMock.getItem.mockReturnValueOnce(JSON.stringify(storedData));

      exportMessages();

      expect(document.createElement).toHaveBeenCalledWith('a');
      expect(mockLink.click).toHaveBeenCalled();
      expect(mockLink.download).toMatch(/james-chat-history-\d{4}-\d{2}-\d{2}\.json/);
    });

    it('should handle export errors gracefully', () => {
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
      localStorageMock.getItem.mockImplementationOnce(() => {
        throw new Error('Export error');
      });

      expect(() => exportMessages()).not.toThrow();
      expect(consoleSpy).toHaveBeenCalledWith(
        'Failed to export messages:',
        expect.any(Error)
      );

      consoleSpy.mockRestore();
    });
  });

  describe('importMessages', () => {
    it('should import valid messages from file', async () => {
      const fileContent = JSON.stringify([
        {
          id: 'imported-1',
          content: 'Imported message',
          isFromUser: true,
          timestamp: '2024-01-01T12:00:00.000Z',
        },
      ]);

      const file = new File([fileContent], 'messages.json', {
        type: 'application/json',
      });

      const messages = await importMessages(file);

      expect(messages).toHaveLength(1);
      expect(messages[0].content).toBe('Imported message');
      expect(messages[0].timestamp).toBeInstanceOf(Date);
    });

    it('should reject invalid file format', async () => {
      const fileContent = 'not json';
      const file = new File([fileContent], 'messages.json', {
        type: 'application/json',
      });

      await expect(importMessages(file)).rejects.toThrow('Failed to parse imported file');
    });

    it('should reject non-array content', async () => {
      const fileContent = JSON.stringify({ not: 'array' });
      const file = new File([fileContent], 'messages.json', {
        type: 'application/json',
      });

      await expect(importMessages(file)).rejects.toThrow('Invalid file format: expected array of messages');
    });

    it('should filter out invalid imported messages', async () => {
      const fileContent = JSON.stringify([
        {
          content: 'Valid message',
          isFromUser: true,
          timestamp: '2024-01-01T12:00:00.000Z',
        },
        {
          // Invalid - missing required fields
          content: 'Invalid message',
        },
      ]);

      const file = new File([fileContent], 'messages.json', {
        type: 'application/json',
      });

      const messages = await importMessages(file);

      expect(messages).toHaveLength(1);
      expect(messages[0].content).toBe('Valid message');
    });
  });

  describe('cleanupOldMessages', () => {
    it('should remove messages older than specified days', () => {
      const oldDate = new Date();
      oldDate.setDate(oldDate.getDate() - 40); // 40 days ago

      const recentDate = new Date();
      recentDate.setDate(recentDate.getDate() - 10); // 10 days ago

      const messages = [
        {
          id: 'old-1',
          content: 'Old message',
          isFromUser: true,
          timestamp: oldDate.toISOString(),
        },
        {
          id: 'recent-1',
          content: 'Recent message',
          isFromUser: true,
          timestamp: recentDate.toISOString(),
        },
      ];

      localStorageMock.getItem.mockReturnValueOnce(JSON.stringify(messages));

      cleanupOldMessages(30); // Keep messages from last 30 days

      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        'james-chat-messages',
        expect.stringContaining('Recent message')
      );
      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        'james-chat-messages',
        expect.not.stringContaining('Old message')
      );
    });

    it('should not modify storage if no cleanup needed', () => {
      const recentMessages = [
        {
          id: 'recent-1',
          content: 'Recent message',
          isFromUser: true,
          timestamp: new Date().toISOString(),
        },
      ];

      localStorageMock.getItem.mockReturnValueOnce(JSON.stringify(recentMessages));

      cleanupOldMessages(30);

      // Should not call setItem since no cleanup was needed
      expect(localStorageMock.setItem).not.toHaveBeenCalled();
    });
  });
});