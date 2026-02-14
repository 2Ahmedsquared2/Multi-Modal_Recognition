import { useState, useEffect } from 'react';

/**
 * Lightweight hook that watches <html class="dark"> via MutationObserver.
 * Components that need to know the current theme (e.g. Plotly charts)
 * can subscribe without importing the full useTheme toggle logic.
 */
export function useDarkMode(): boolean {
  const [isDark, setIsDark] = useState(
    () => document.documentElement.classList.contains('dark'),
  );

  useEffect(() => {
    const observer = new MutationObserver(() => {
      setIsDark(document.documentElement.classList.contains('dark'));
    });
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class'],
    });
    return () => observer.disconnect();
  }, []);

  return isDark;
}
