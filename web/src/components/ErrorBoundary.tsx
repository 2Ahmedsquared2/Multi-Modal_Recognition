import { Component, type ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('ErrorBoundary caught:', error, info.componentStack);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="max-w-2xl mx-auto px-8 py-20 text-center space-y-4">
          <div className="w-12 h-12 mx-auto rounded-xl bg-rose-50 dark:bg-rose-500/10 flex items-center justify-center">
            <svg className="w-6 h-6 text-rose-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <path d="M12 8v4M12 16h.01" />
            </svg>
          </div>
          <h2 className="text-lg font-semibold text-warm-900 dark:text-warm-100">
            Something went wrong
          </h2>
          <pre className="text-left text-xs font-mono bg-warm-200 dark:bg-warm-700 p-4 rounded-lg overflow-auto text-rose-600 dark:text-rose-400 max-h-40">
            {this.state.error?.message}
            {'\n'}
            {this.state.error?.stack}
          </pre>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            className="px-4 py-2 text-sm font-medium rounded-lg bg-warm-200 dark:bg-warm-700 text-warm-700 dark:text-warm-300 hover:bg-warm-300 dark:hover:bg-warm-600 transition-colors"
          >
            Try again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
