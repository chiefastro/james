import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { CopilotProvider } from '@/components/CopilotProvider';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Conscious Agent System',
  description: 'A conscious AI agent system with stream-of-consciousness capabilities',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <CopilotProvider>
          {children}
        </CopilotProvider>
      </body>
    </html>
  );
}