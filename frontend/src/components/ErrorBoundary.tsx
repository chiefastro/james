/**
 * Error boundary component for handling React errors gracefully.
 * 
 * This component catches JavaScript errors anywhere in the child component tree,
 * logs those errors, and displays a fallback UI instead of crashing the app.
 */

"use client";

import React, { Component, ReactNode } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface ErrorBoundaryProps {
  children: ReactNode;
  maxRetries?: number;
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: React.ErrorInfo | null;
  retryCount: number;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      retryCount: 0,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return {
      hasError: true,
      error,
    };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    this.setState({
      error,
      errorInfo,
    });

    // Call the onError callback if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // Log error to console in development
    if (process.env.NODE_ENV === 'development') {
      console.error('ErrorBoundary caught an error:', error, errorInfo);
    }
  }

  handleRetry = () => {
    const { maxRetries = 3 } = this.props;
    const { retryCount } = this.state;

    if (retryCount < maxRetries) {
      this.setState({
        hasError: false,
        error: null,
        errorInfo: null,
        retryCount: retryCount + 1,
      });
    }
  };

  handleReload = () => {
    window.location.reload();
  };

  render() {
    const { hasError, error, retryCount } = this.state;
    const { maxRetries = 3 } = this.props;

    if (hasError) {
      return (
        <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
          <div className="max-w-md w-full bg-white rounded-lg shadow-md p-6 text-center">
            <AlertTriangle className="h-12 w-12 text-red-500 mx-auto mb-4" />
            
            <h2 className="text-xl font-semibold text-gray-900 mb-2">
              Something went wrong
            </h2>
            
            <p className="text-gray-600 mb-6">
              We encountered an unexpected error. This might be a temporary issue.
            </p>

            {process.env.NODE_ENV === 'development' && error && (
              <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-md text-left">
                <h3 className="text-sm font-medium text-red-800 mb-2">Error Details:</h3>
                <pre className="text-xs text-red-700 whitespace-pre-wrap overflow-auto max-h-32">
                  {error.toString()}
                </pre>
              </div>
            )}

            <div className="space-y-3">
              {retryCount < maxRetries && (
                <button
                  onClick={this.handleRetry}
                  className="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors flex items-center justify-center"
                >
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Try Again ({maxRetries - retryCount} attempts left)
                </button>
              )}
              
              <button
                onClick={this.handleReload}
                className="w-full px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 transition-colors"
              >
                Reload Page
              </button>
            </div>

            {retryCount >= maxRetries && (
              <p className="text-sm text-gray-500 mt-4">
                If the problem persists, please try reloading the page or contact support.
              </p>
            )}
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}