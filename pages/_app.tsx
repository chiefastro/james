import '@/styles/globals.css'
import type { AppProps } from 'next/app'
import { CopilotKit } from '@copilotkit/react-core'

export default function App({ Component, pageProps }: AppProps) {
  return (
    <CopilotKit runtimeUrl="http://localhost:8000">
      <Component {...pageProps} />
    </CopilotKit>
  )
}