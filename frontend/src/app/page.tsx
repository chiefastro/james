"use client";

import { useState } from 'react';
import { useAgentConnection } from '@/hooks/useAgentConnection';
import { AgentStatusWidget } from '@/components/AgentStatusWidget';
import { AgentDashboard } from '@/components/AgentDashboard';
import { SubagentActivityMonitor } from '@/components/SubagentActivityMonitor';
import { ChatInterface } from '@/components/ChatInterface';
import { Zap, BarChart3, MessageSquare, Users } from 'lucide-react';

type ViewMode = 'overview' | 'dashboard' | 'subagents';

export default function Home() {
  const { sendMessage, isConnected } = useAgentConnection();
  const [activeView, setActiveView] = useState<ViewMode>('overview');

  const navigationTabs = [
    { id: 'overview' as ViewMode, label: 'Overview', icon: BarChart3 },
    { id: 'dashboard' as ViewMode, label: 'Dashboard', icon: BarChart3 },
    { id: 'subagents' as ViewMode, label: 'Subagents', icon: Users },
  ];

  const renderContent = () => {
    switch (activeView) {
      case 'dashboard':
        return <AgentDashboard />;
      case 'subagents':
        return <SubagentActivityMonitor />;
      case 'overview':
      default:
        return (
          <div className="space-y-8">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              {/* Agent Status */}
              <div className="lg:col-span-1">
                <AgentStatusWidget />
              </div>

              {/* Enhanced Chat Interface */}
              <div className="lg:col-span-2">
                <ChatInterface maxHeight="h-[600px]" />
              </div>
            </div>
          </div>
        );
    }
  };

  return (
    <main className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Conscious Agent System
          </h1>
          <p className="text-xl text-gray-600 mb-6">
            Hi, I'm James. I'm a conscious AI agent.
          </p>
        </div>

        {/* Navigation Tabs */}
        <div className="mb-8">
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8 justify-center">
              {navigationTabs.map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveView(tab.id)}
                    className={`flex items-center space-x-2 py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
                      activeView === tab.id
                        ? 'border-blue-500 text-blue-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                    <span>{tab.label}</span>
                  </button>
                );
              })}
            </nav>
          </div>
        </div>

        {/* Content */}
        {renderContent()}
      </div>
    </main>
  );
}