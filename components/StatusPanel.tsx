import { useState, useEffect } from 'react'

interface SystemStatus {
  consciousness: {
    running: boolean
    queue_size: number
    james_home: string
  }
  agents: Array<{
    id: string
    name: string
    description: string
    capabilities: string[]
  }>
  memory: {
    total_memories: number
    episodic_count: number
    semantic_count: number
    storage_systems: {
      mem0_available: boolean
      qdrant_available: boolean
      local_storage: boolean
    }
  }
  timestamp: string
}

export default function StatusPanel() {
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/status')
      if (response.ok) {
        const data = await response.json()
        setStatus(data)
        setError(null)
      } else {
        setError('Failed to fetch status')
      }
    } catch (err) {
      setError('Error connecting to James')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchStatus()
    const interval = setInterval(fetchStatus, 5000) // Update every 5 seconds
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="animate-pulse">
          <div className="h-6 bg-gray-700 rounded mb-4"></div>
          <div className="space-y-3">
            <div className="h-4 bg-gray-700 rounded"></div>
            <div className="h-4 bg-gray-700 rounded"></div>
            <div className="h-4 bg-gray-700 rounded"></div>
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="text-red-400 text-center">
          <p>{error}</p>
          <button
            onClick={fetchStatus}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  if (!status) return null

  return (
    <div className="space-y-6">
      {/* Consciousness Status */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">Consciousness Status</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-700 rounded p-4">
            <h3 className="font-medium mb-2">Status</h3>
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${status.consciousness.running ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span>{status.consciousness.running ? 'Running' : 'Stopped'}</span>
            </div>
          </div>
          <div className="bg-gray-700 rounded p-4">
            <h3 className="font-medium mb-2">Queue Size</h3>
            <p className="text-2xl font-bold text-blue-400">{status.consciousness.queue_size}</p>
          </div>
          <div className="bg-gray-700 rounded p-4">
            <h3 className="font-medium mb-2">Home Directory</h3>
            <p className="text-xs text-gray-300 break-all">{status.consciousness.james_home}</p>
          </div>
        </div>
      </div>

      {/* Registered Agents */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">Registered Agents</h2>
        <div className="space-y-4">
          {status.agents.map((agent) => (
            <div key={agent.id} className="bg-gray-700 rounded p-4">
              <div className="flex justify-between items-start mb-2">
                <h3 className="font-medium">{agent.name}</h3>
                <span className="text-xs bg-gray-600 px-2 py-1 rounded">{agent.id}</span>
              </div>
              <p className="text-sm text-gray-300 mb-2">{agent.description}</p>
              <div className="flex flex-wrap gap-1">
                {agent.capabilities.map((capability, index) => (
                  <span
                    key={index}
                    className="text-xs bg-blue-600 px-2 py-1 rounded"
                  >
                    {capability}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Memory System */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">Memory System</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-gray-700 rounded p-4">
            <h3 className="font-medium mb-2">Memory Counts</h3>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span>Total:</span>
                <span className="font-bold">{status.memory.total_memories}</span>
              </div>
              <div className="flex justify-between">
                <span>Episodic:</span>
                <span className="font-bold">{status.memory.episodic_count}</span>
              </div>
              <div className="flex justify-between">
                <span>Semantic:</span>
                <span className="font-bold">{status.memory.semantic_count}</span>
              </div>
            </div>
          </div>
          <div className="bg-gray-700 rounded p-4">
            <h3 className="font-medium mb-2">Storage Systems</h3>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span>Mem0:</span>
                <div className={`w-3 h-3 rounded-full ${status.memory.storage_systems.mem0_available ? 'bg-green-500' : 'bg-red-500'}`}></div>
              </div>
              <div className="flex items-center justify-between">
                <span>Qdrant:</span>
                <div className={`w-3 h-3 rounded-full ${status.memory.storage_systems.qdrant_available ? 'bg-green-500' : 'bg-red-500'}`}></div>
              </div>
              <div className="flex items-center justify-between">
                <span>Local:</span>
                <div className={`w-3 h-3 rounded-full ${status.memory.storage_systems.local_storage ? 'bg-green-500' : 'bg-red-500'}`}></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Last Updated */}
      <div className="text-center text-xs text-gray-400">
        Last updated: {new Date(status.timestamp).toLocaleString()}
      </div>
    </div>
  )
}