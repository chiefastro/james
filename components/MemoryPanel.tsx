import { useState } from 'react'

interface Memory {
  content: string
  score: number
  source: string
  type: string
  metadata: Record<string, any>
  timestamp: string
}

export default function MemoryPanel() {
  const [query, setQuery] = useState('')
  const [memoryType, setMemoryType] = useState<'episodic' | 'semantic' | ''>('')
  const [results, setResults] = useState<Memory[]>([])
  const [loading, setLoading] = useState(false)
  const [newMemory, setNewMemory] = useState('')
  const [newMemoryType, setNewMemoryType] = useState<'episodic' | 'semantic'>('episodic')

  const searchMemories = async () => {
    if (!query.trim()) return

    setLoading(true)
    try {
      const response = await fetch('http://localhost:8000/memory/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          memory_type: memoryType || undefined,
          limit: 20
        }),
      })

      if (response.ok) {
        const data = await response.json()
        setResults(data.results)
      } else {
        console.error('Failed to search memories')
      }
    } catch (error) {
      console.error('Error searching memories:', error)
    } finally {
      setLoading(false)
    }
  }

  const storeMemory = async () => {
    if (!newMemory.trim()) return

    try {
      const response = await fetch('http://localhost:8000/memory/store', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content: newMemory,
          memory_type: newMemoryType,
          metadata: { source: 'ui', created_by: 'user' }
        }),
      })

      if (response.ok) {
        setNewMemory('')
        alert('Memory stored successfully!')
      } else {
        console.error('Failed to store memory')
      }
    } catch (error) {
      console.error('Error storing memory:', error)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      searchMemories()
    }
  }

  return (
    <div className="space-y-6">
      {/* Store New Memory */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">Store New Memory</h2>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">Memory Type</label>
            <select
              value={newMemoryType}
              onChange={(e) => setNewMemoryType(e.target.value as 'episodic' | 'semantic')}
              className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
            >
              <option value="episodic">Episodic (Events/Experiences)</option>
              <option value="semantic">Semantic (Facts/Knowledge)</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Memory Content</label>
            <textarea
              value={newMemory}
              onChange={(e) => setNewMemory(e.target.value)}
              placeholder="Enter memory content..."
              className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white placeholder-gray-400 resize-none"
              rows={3}
            />
          </div>
          <button
            onClick={storeMemory}
            disabled={!newMemory.trim()}
            className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Store Memory
          </button>
        </div>
      </div>

      {/* Search Memories */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">Search Memories</h2>
        <div className="space-y-4">
          <div className="flex space-x-4">
            <div className="flex-1">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Search memories..."
                className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white placeholder-gray-400"
              />
            </div>
            <div>
              <select
                value={memoryType}
                onChange={(e) => setMemoryType(e.target.value as 'episodic' | 'semantic' | '')}
                className="bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
              >
                <option value="">All Types</option>
                <option value="episodic">Episodic</option>
                <option value="semantic">Semantic</option>
              </select>
            </div>
            <button
              onClick={searchMemories}
              disabled={!query.trim() || loading}
              className="px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Searching...' : 'Search'}
            </button>
          </div>

          {/* Results */}
          <div className="space-y-3">
            {results.length === 0 && query && !loading && (
              <p className="text-gray-400 text-center py-8">No memories found for "{query}"</p>
            )}
            
            {results.map((memory, index) => (
              <div key={index} className="bg-gray-700 rounded p-4">
                <div className="flex justify-between items-start mb-2">
                  <div className="flex space-x-2">
                    <span className={`text-xs px-2 py-1 rounded ${
                      memory.type === 'episodic' ? 'bg-purple-600' : 'bg-orange-600'
                    }`}>
                      {memory.type}
                    </span>
                    <span className="text-xs px-2 py-1 rounded bg-gray-600">
                      {memory.source}
                    </span>
                  </div>
                  <span className="text-xs text-gray-400">
                    Score: {memory.score.toFixed(2)}
                  </span>
                </div>
                <p className="text-gray-100 mb-2">{memory.content}</p>
                {memory.timestamp && (
                  <p className="text-xs text-gray-400">
                    {new Date(memory.timestamp).toLocaleString()}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}