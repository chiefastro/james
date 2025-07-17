// Mock for @copilotkit/react-core
module.exports = {
  useCopilotAction: jest.fn(() => ({})),
  useCopilotReadable: jest.fn(() => ({})),
  useCopilotChat: jest.fn(() => ({
    messages: [],
    appendMessage: jest.fn(),
    setMessages: jest.fn(),
    deleteMessage: jest.fn(),
    reloadMessages: jest.fn(),
    stopGeneration: jest.fn(),
    isLoading: false,
  })),
  CopilotProvider: ({ children }) => children,
};