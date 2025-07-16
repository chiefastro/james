/**
 * TypeScript interfaces matching Python models for frontend type safety.
 * 
 * These interfaces provide type safety for frontend-backend communication
 * and ensure consistency with the Python data models.
 */

// Enums matching Python enums
export enum MessageSource {
  USER = "user",
  SUBAGENT = "subagent",
  SYSTEM = "system",
  EXTERNAL = "external",
}

export enum MessageClassification {
  IGNORE_DELETE = "ignore_delete",
  DELAY = "delay",
  ARCHIVE = "archive",
  ACT_NOW = "act_now",
}

export enum TaskStatus {
  PENDING = "pending",
  IN_PROGRESS = "in_progress",
  COMPLETED = "completed",
  FAILED = "failed",
  CANCELLED = "cancelled",
}

// Core data interfaces
export interface Message {
  id: string;
  content: string;
  source: MessageSource;
  priority: number;
  timestamp: string; // ISO datetime string
  metadata: Record<string, any>;
  classification?: MessageClassification | null;
  delay_seconds?: number | null;
}

export interface Subagent {
  id: string;
  name: string;
  description: string;
  input_schema: Record<string, any>;
  output_schema: Record<string, any>;
  import_path: string;
  embedding: number[];
  capabilities: string[];
  created_at: string; // ISO datetime string
  last_used?: string | null; // ISO datetime string
  is_active: boolean;
}

export interface Task {
  id: string;
  description: string;
  priority: number;
  status: TaskStatus;
  assigned_subagents: string[];
  created_at: string; // ISO datetime string
  updated_at: string; // ISO datetime string
  deadline?: string | null; // ISO datetime string
  parent_task_id?: string | null;
  result?: Record<string, any> | null;
  error_message?: string | null;
}

// API Request interfaces
export interface MessageRequest {
  content: string;
  source?: MessageSource;
  priority?: number;
  metadata?: Record<string, any>;
}

export interface SubagentRequest {
  name: string;
  description: string;
  input_schema?: Record<string, any>;
  output_schema?: Record<string, any>;
  import_path: string;
  capabilities?: string[];
  is_active?: boolean;
}

export interface TaskRequest {
  description: string;
  priority?: number;
  deadline?: string | null; // ISO datetime string
  parent_task_id?: string | null;
}

// API Response interfaces
export interface MessageResponse extends Message {}

export interface SubagentResponse extends Omit<Subagent, 'embedding'> {
  // Exclude embedding from response for performance
}

export interface TaskResponse extends Task {}

export interface AgentStatusResponse {
  is_active: boolean;
  current_tasks: TaskResponse[];
  message_queue_size: number;
  active_subagents: number;
  memory_usage: Record<string, any>;
  uptime_seconds: number;
  last_activity?: string | null; // ISO datetime string
}

export interface ErrorResponse {
  error: string;
  message: string;
  details?: Record<string, any> | null;
  timestamp: string; // ISO datetime string
}

export interface HealthResponse {
  status: string;
  timestamp: string; // ISO datetime string
  version: string;
  components: Record<string, string>;
}

// Utility types for frontend use
export interface ChatMessage {
  id: string;
  content: string;
  isFromUser: boolean;
  timestamp: Date;
  status?: 'sending' | 'sent' | 'error';
}

export interface AgentState {
  isConnected: boolean;
  isTyping: boolean;
  currentTasks: Task[];
  queueSize: number;
  lastActivity?: Date;
}

// WebSocket message types
export interface WebSocketMessage {
  type: 'message' | 'status' | 'error' | 'ping' | 'pong';
  data: any;
  timestamp: string;
}

// Form validation types
export type ValidationError = {
  field: string;
  message: string;
};

export type FormErrors<T> = {
  [K in keyof T]?: string;
};

// API response wrapper
export interface ApiResponse<T = any> {
  data?: T;
  error?: ErrorResponse;
  success: boolean;
}