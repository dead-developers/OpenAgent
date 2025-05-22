import { useState, useEffect, useRef, useCallback } from 'react';

// WebSocket message type
interface WebSocketMessage {
  type: string;
  timestamp: string;
  execution_id?: string;
  data: any;
}

// Connection status type
type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

/**
 * Hook for WebSocket connection
 * @param clientId Client ID for WebSocket connection
 * @param executionId Optional execution ID to filter messages
 * @returns WebSocket connection status and last message
 */
export const useWebSocket = (clientId: string, executionId?: string) => {
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Connect to WebSocket
  const connect = useCallback(() => {
    // Clear any existing reconnect timeout
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    // Close existing socket if any
    if (socketRef.current) {
      socketRef.current.close();
    }

    // Set connection status to connecting
    setConnectionStatus('connecting');

    // Create new WebSocket connection
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/api/ws/connect/${clientId}`;
    
    const socket = new WebSocket(wsUrl);

    // Set up event handlers
    socket.onopen = () => {
      console.log('WebSocket connected');
      setConnectionStatus('connected');
      
      // Send ping every 30 seconds to keep connection alive
      const pingInterval = setInterval(() => {
        if (socket.readyState === WebSocket.OPEN) {
          socket.send(JSON.stringify({ type: 'ping' }));
        }
      }, 30000);
      
      // Clear ping interval when socket is closed
      socket.onclose = () => {
        clearInterval(pingInterval);
        setConnectionStatus('disconnected');
        console.log('WebSocket disconnected');
        
        // Reconnect after 5 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          console.log('Reconnecting WebSocket...');
          connect();
        }, 5000);
      };
    };

    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnectionStatus('error');
    };

    socket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data) as WebSocketMessage;
        
        // Filter messages by execution ID if provided
        if (!executionId || !message.execution_id || message.execution_id === executionId) {
          setLastMessage(message);
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    socketRef.current = socket;

    // Clean up on unmount
    return () => {
      if (socket) {
        socket.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [clientId, executionId]);

  // Connect on mount
  useEffect(() => {
    const cleanup = connect();
    
    // Clean up on unmount
    return cleanup;
  }, [connect]);

  // Send message
  const sendMessage = useCallback((message: any) => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify(message));
    } else {
      console.error('WebSocket not connected');
    }
  }, []);

  return {
    lastMessage,
    connectionStatus,
    sendMessage
  };
};
