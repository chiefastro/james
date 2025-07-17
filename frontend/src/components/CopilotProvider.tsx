/**
 * CopilotKit provider component for the Conscious Agent System.
 * 
 * This component wraps the application with CopilotKit providers and
 * configures the integration with the conscious agent backend.
 */

"use client";

import { CopilotKit } from "@copilotkit/react-core";
import { CopilotPopup } from "@copilotkit/react-ui";
import { ReactNode } from "react";
import { copilotConfig } from "@/lib/copilot-config";
import { ErrorBoundary } from "./ErrorBoundary";

interface CopilotProviderProps {
  children: ReactNode;
}

export function CopilotProvider({ children }: CopilotProviderProps) {
  return (
    <ErrorBoundary
      maxRetries={3}
      onError={(error, errorInfo) => {
        console.error('CopilotKit Error:', error, errorInfo);
        // Could send to error reporting service here
      }}
    >
      <CopilotKit 
        runtimeUrl={copilotConfig.runtimeUrl}
        agent={copilotConfig.agent.name}
      >
        {children}
      </CopilotKit>
    </ErrorBoundary>
  );
}