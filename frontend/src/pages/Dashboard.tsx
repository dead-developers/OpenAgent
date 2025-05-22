import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Grid, 
  Paper, 
  Typography, 
  Box, 
  TextField, 
  Button,
  CircularProgress,
  Alert,
  Snackbar,
  Card,
  CardContent,
  Divider,
  Chip
} from '@mui/material';
import { Send as SendIcon } from '@mui/icons-material';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

import { useWebSocket } from '../hooks/useWebSocket';
import PlanVisualizer from '../components/PlanVisualizer';
import ToolExecutionVisualizer from '../components/ToolExecutionVisualizer';

// Types
interface Execution {
  id: string;
  prompt: string;
  result?: string;
  status: 'running' | 'completed' | 'error';
  start_time: string;
  end_time?: string;
  agent_type?: string;
}

const Dashboard: React.FC = () => {
  const [prompt, setPrompt] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [currentExecution, setCurrentExecution] = useState<Execution | null>(null);
  const [agentType, setAgentType] = useState('manus');
  const [usePlanning, setUsePlanning] = useState(true);
  const navigate = useNavigate();
  
  // Generate client ID for WebSocket
  const clientId = React.useMemo(() => `client_${Math.random().toString(36).substring(2, 15)}`, []);
  
  // Connect to WebSocket
  const { lastMessage, connectionStatus } = useWebSocket(clientId);
  
  // Process WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      const { type, execution_id, data } = lastMessage;
      
      // Only process messages for current execution
      if (currentExecution && execution_id === currentExecution.id) {
        switch (type) {
          case 'execution_completed':
            setCurrentExecution(prev => prev ? {
              ...prev,
              status: 'completed',
              result: data.execution.result,
              end_time: data.execution.end_time
            } : null);
            setSuccess('Execution completed successfully');
            break;
            
          case 'execution_error':
            setCurrentExecution(prev => prev ? {
              ...prev,
              status: 'error',
              result: data.execution.error,
              end_time: data.execution.end_time
            } : null);
            setError(`Execution failed: ${data.execution.error}`);
            break;
            
          case 'thinking_update':
            // Update thinking content
            break;
        }
      }
    }
  }, [lastMessage, currentExecution]);
  
  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;
    
    setIsSubmitting(true);
    setError(null);
    
    try {
      // Start execution
      const response = await axios.post('/api/executions', {
        prompt,
        use_planning: usePlanning,
        agent_type: agentType
      });
      
      // Set current execution
      setCurrentExecution(response.data);
      
      // Clear prompt
      setPrompt('');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to start execution');
    } finally {
      setIsSubmitting(false);
    }
  };
  
  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      {/* Error and success messages */}
      <Snackbar 
        open={!!error} 
        autoHideDuration={6000} 
        onClose={() => setError(null)}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert onClose={() => setError(null)} severity="error" sx={{ width: '100%' }}>
          {error}
        </Alert>
      </Snackbar>
      
      <Snackbar 
        open={!!success} 
        autoHideDuration={6000} 
        onClose={() => setSuccess(null)}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert onClose={() => setSuccess(null)} severity="success" sx={{ width: '100%' }}>
          {success}
        </Alert>
      </Snackbar>
      
      <Grid container spacing={3}>
        {/* Prompt input */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <form onSubmit={handleSubmit}>
              <Typography variant="h6" gutterBottom>
                Ask OpenAgent
              </Typography>
              <TextField
                fullWidth
                multiline
                rows={4}
                label="Enter your prompt"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="What would you like OpenAgent to do?"
                variant="outlined"
                margin="normal"
                disabled={isSubmitting}
              />
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 2 }}>
                <Box>
                  <Chip 
                    label={`Agent: ${agentType}`}
                    color="primary"
                    variant="outlined"
                    sx={{ mr: 1 }}
                    onClick={() => setAgentType(agentType === 'manus' ? 'mcp' : 'manus')}
                  />
                  <Chip 
                    label={`Planning: ${usePlanning ? 'On' : 'Off'}`}
                    color={usePlanning ? 'success' : 'default'}
                    variant="outlined"
                    onClick={() => setUsePlanning(!usePlanning)}
                  />
                </Box>
                <Button 
                  type="submit" 
                  variant="contained" 
                  color="primary"
                  endIcon={<SendIcon />}
                  disabled={isSubmitting || !prompt.trim()}
                >
                  {isSubmitting ? 'Processing...' : 'Submit'}
                </Button>
              </Box>
            </form>
          </Paper>
        </Grid>
        
        {/* Execution visualizers */}
        {currentExecution && (
          <>
            <Grid item xs={12}>
              <Paper sx={{ p: 2, mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Typography variant="h6">
                    Execution Status: {currentExecution.status}
                  </Typography>
                  <Chip 
                    label={`ID: ${currentExecution.id}`}
                    variant="outlined"
                    size="small"
                  />
                </Box>
                <Divider sx={{ my: 1 }} />
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Prompt: {currentExecution.prompt}
                </Typography>
                {currentExecution.status === 'running' && (
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                    <CircularProgress size={20} sx={{ mr: 1 }} />
                    <Typography variant="body2">Processing your request...</Typography>
                  </Box>
                )}
                {currentExecution.result && (
                  <Box sx={{ mt: 2, p: 2, bgcolor: 'background.default', borderRadius: 1 }}>
                    <Typography variant="body1" component="pre" sx={{ whiteSpace: 'pre-wrap' }}>
                      {currentExecution.result}
                    </Typography>
                  </Box>
                )}
              </Paper>
            </Grid>
            
            {/* Plan visualizer */}
            <Grid item xs={12} md={6}>
              <PlanVisualizer executionId={currentExecution.id} />
            </Grid>
            
            {/* Tool execution visualizer */}
            <Grid item xs={12} md={6}>
              <ToolExecutionVisualizer executionId={currentExecution.id} />
            </Grid>
          </>
        )}
        
        {/* Welcome card when no execution is running */}
        {!currentExecution && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h5" gutterBottom>
                  Welcome to OpenAgent UI
                </Typography>
                <Typography variant="body1" paragraph>
                  OpenAgent is a dual-model AI agent system designed to leverage multiple language models for enhanced reasoning, problem-solving, and autonomous task execution.
                </Typography>
                <Typography variant="body1">
                  Enter a prompt above to get started, or explore previous executions in the <Button variant="text" onClick={() => navigate('/history')}>History</Button> page.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Container>
  );
};

export default Dashboard;
