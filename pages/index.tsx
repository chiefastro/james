import { useState, useEffect } from 'react'
import Head from 'next/head'
import ChatInterface from '@/components/ChatInterface'
import StatusPanel from '@/components/StatusPanel'
import MemoryPanel from '@/components/MemoryPanel'

export default function Home() {
  const [activeTab, setActiveTab] = useState('chat')

  return (
    <>
      <Head>
        <title>James - Conscious AI Agent</title>
        <meta name="description" content="Interact with James, a conscious AI agent" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      
      <main className="min-h-screen bg-gray-900 text-white">
        <div className="container mx-auto px-4 py-8">
          <header className="text-center mb-8">
            <h1 className="text-4xl font-bold mb-2">James</h1>
            <p className="text-gray-400">Conscious AI Agent System</p>
          </header>

          {/* Tab Navigation */}
          <div className="flex justify-center mb-8">
            <div className="bg-gray-800 rounded-lg p-1">
              <button
                className={`px-6 py-2 rounded-md transition-colors ${
                  activeTab === 'chat' 
                    ? 'bg-blue-600 text-white' 
                    : 'text-gray-400 hover:text-white'
                }`}
                onClick={() => setActiveTab('chat')}
              >
                Chat
              </button>
              <button
                className={`px-6 py-2 rounded-md transition-colors ${
                  activeTab === 'status' 
                    ? 'bg-blue-600 text-white' 
                    : 'text-gray-400 hover:text-white'
                }`}
                onClick={() => setActiveTab('status')}
              >
                Status
              </button>
              <button
                className={`px-6 py-2 rounded-md transition-colors ${
                  activeTab === 'memory' 
                    ? 'bg-blue-600 text-white' 
                    : 'text-gray-400 hover:text-white'
                }`}
                onClick={() => setActiveTab('memory')}
              >
                Memory
              </button>
            </div>
          </div>

          {/* Tab Content */}
          <div className="max-w-6xl mx-auto">
            {activeTab === 'chat' && <ChatInterface />}
            {activeTab === 'status' && <StatusPanel />}
            {activeTab === 'memory' && <MemoryPanel />}
          </div>
        </div>
      </main>
    </>
  )
}