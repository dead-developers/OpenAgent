import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Card, 
  CardContent, 
  Typography, 
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Collapse,
  IconButton,
  Paper,
  Divider,
  Chip,
  Tooltip,
  CircularProgress
} from '@mui/material';
import { 
  Code as CodeIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Input as InputIcon,
  Output as OutputIcon
} from '@mui/icons-material';
import axios from 'axios';
import { useWebSocket } from '../hooks/useWebSocket';

// Types
interface ToolExecution {
  id: string;
  execution_id: string;
  step_number: number;
  tool_name: string;
  input_data: any;
  output_data: any;
  status: string;
  execution_time: number;
  timestamp: string;
}

interface ToolExecutionVisualizerProps {
  executionId: string;
  initialExecutions?: ToolExecution[];
}

// Tool execution visualizer component
const ToolExecutionVisualizer: React.FC<ToolExecutionVisualizerProps> = ({ 
  executionId, 
  initialExecutions = [] 
}) => {
  const [executions, setExecutions] = useState<ToolExecution[]>(initialExecutions);
  const [expandedItems, setExpandedItems] = useState<Record<string, boolean>>({});
  const [loading, setLoading] = useState<boolean>(!initialExecutions.length);
  const [error, setError] = useState<string | null>(null);
  
  // Connect to WebSocket for real-time updates
  const { lastMessage } = useWebSocket(executionId);
  
  // Fetch initial executions if not provided
  useEffect(() => {
    const fetchExecutions = async () => {
      if (!initialExecutions.length) {
        setLoading(true);
        try {
          const response = await axios.get(`/api/executions/${executionId}/steps`);
          setExecutions(response.data);
          
          // Auto-expand the first execution
          if (response.data.length > 0) {
            setExpandedItems(prev => ({
              ...prev,
              [response.data[0].id]: true
            }));
          }
        } catch (err) {
          console.error('Error fetching tool executions:', err);
          setError('Failed to fetch tool executions');
        } finally {
          setLoading(false);
        }
      }
    };
    
    fetchExecutions();
  }, [executionId, initialExecutions]);
  
  // Process WebSocket messages
  useEffect(() => {
    if (lastMessage && lastMessage.type === 'step_update' && lastMessage.execution_id === executionId) {
      const newExecution = lastMessage.data.step;
      
      // Check if this execution already exists
      const existingIndex = executions.findIndex(exec => exec.id === newExecution.id);
      
      if (existingIndex === -1) {
        // Add new execution
        setExecutions(prev => [...prev, newExecution]);
        
        // Auto-expand the newest execution
        setExpandedItems(prev => ({
          ...prev,
          [newExecution.id]: true
        }));
      } else {
        // Update existing execution
        setExecutions(prev => prev.map(exec => 
          exec.id === newExecution.id ? newExecution : exec
        ));
      }
    }
  }, [lastMessage, executionId, executions]);
  
  // Toggle item expansion
  const toggleItemExpansion = (itemId: string) => {
    setExpandedItems(prev => ({
      ...prev,
      [itemId]: !prev[itemId]
    }));
  };
  
  if (loading) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6">Loading tool executions...</Typography>
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 3 }}>
            <CircularProgress />
          </Box>
        </CardContent>
      </Card>
    );
  }
  
  if (error) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" color="error">Error</Typography>
          <Typography variant="body2">{error}</Typography>
        </CardContent>
      </Card>
    );
  }
  
  if (executions.length === 0) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6">No tool executions yet</Typography>
          <Typography variant="body2">
            Tool executions will appear here as they are performed.
          </Typography>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>Tool Executions</Typography>
        
        <List>
          {executions.map((execution) => (
            <React.Fragment key={execution.id}>
              <ListItem 
                button 
                onClick={() => toggleItemExpansion(execution.id)}
                sx={{ 
                  bgcolor: 'background.paper', 
                  mb: 1, 
                  borderRadius: 1,
                  border: '1px solid',
                  borderColor: 'divider'
                }}
              >
                <ListItemIcon>
                  <CodeIcon />
                </ListItemIcon>
                <ListItemText 
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Typography variant="subtitle1">{execution.tool_name}</Typography>
                      <Chip 
                        label={`Step ${execution.step_number}`} 
                        size="small" 
                        sx={{ ml: 1 }} 
                        color="primary"
                      />
                      {execution.status && (
                        <Chip 
                          label={execution.status} 
                          size="small" 
                          sx={{ ml: 1 }} 
                          color={execution.status === 'success' ? 'success' : 'error'}
                        />
                      )}
                    </Box>
                  }
                  secondary={new Date(execution.timestamp).toLocaleString()}
                />
                <Box sx={{ mr: 1 }}>
                  <Tooltip title={`Execution time: ${execution.execution_time}ms`}>
                    <Chip 
                      label={`${execution.execution_time}ms`} 
                      size="small" 
                      variant="outlined"
                    />
                  </Tooltip>
                </Box>
                <IconButton edge="end">
                  {expandedItems[execution.id] ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                </IconButton>
              </ListItem>
              
              <Collapse in={expandedItems[execution.id]} timeout="auto" unmountOnExit>
                <Paper sx={{ p: 2, mb: 2, ml: 4 }}>
                  <Typography variant="subtitle2" sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <InputIcon fontSize="small" sx={{ mr: 1 }} />
                    Input
                  </Typography>
                  <Box sx={{ mb: 2, p: 1, bgcolor: 'action.hover', borderRadius: 1, overflow: 'auto' }}>
                    <pre style={{ margin: 0 }}>
                      {JSON.stringify(execution.input_data, null, 2)}
                    </pre>
                  </Box>
                  
                  <Divider sx={{ my: 2 }} />
                  
                  <Typography variant="subtitle2" sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <OutputIcon fontSize="small" sx={{ mr: 1 }} />
                    Output
                  </Typography>
                  <Box sx={{ p: 1, bgcolor: 'action.hover', borderRadius: 1, overflow: 'auto' }}>
                    <pre style={{ margin: 0 }}>
                      {JSON.stringify(execution.output_data, null, 2)}
                    </pre>
                  </Box>
                </Paper>
              </Collapse>
            </React.Fragment>
          ))}
        </List>
      </CardContent>
    </Card>
  );
};

export default ToolExecutionVisualizer;
