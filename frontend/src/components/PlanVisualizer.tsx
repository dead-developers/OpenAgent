import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Card, 
  CardContent, 
  Typography, 
  Stepper, 
  Step, 
  StepLabel, 
  StepContent,
  Button,
  Collapse,
  LinearProgress,
  Chip,
  Divider,
  IconButton,
  Tooltip,
  CircularProgress
} from '@mui/material';
import { 
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Pending as PendingIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Info as InfoIcon
} from '@mui/icons-material';
import axios from 'axios';
import { useWebSocket } from '../hooks/useWebSocket';

// Types
interface PlanStep {
  id: string;
  title: string;
  description: string;
  status: 'pending' | 'in_progress' | 'completed' | 'error';
  notes?: string;
}

interface Plan {
  id: string;
  execution_id: string;
  title: string;
  steps: string[];
  step_statuses: string[];
  step_notes: string[];
  progress: number;
  created_at: string;
  updated_at: string;
}

interface PlanVisualizerProps {
  executionId: string;
  initialPlan?: Plan;
}

// Status icon component
const StatusIcon: React.FC<{ status: string }> = ({ status }) => {
  switch (status) {
    case 'completed':
      return <CheckCircleIcon color="success" />;
    case 'error':
      return <ErrorIcon color="error" />;
    case 'in_progress':
      return <PendingIcon color="primary" sx={{ animation: 'spin 2s linear infinite' }} />;
    default:
      return <PendingIcon color="disabled" />;
  }
};

// Plan visualizer component
const PlanVisualizer: React.FC<PlanVisualizerProps> = ({ executionId, initialPlan }) => {
  const [plan, setPlan] = useState<Plan | null>(initialPlan || null);
  const [expandedSteps, setExpandedSteps] = useState<Record<string, boolean>>({});
  const [activeStep, setActiveStep] = useState<number>(-1);
  const [loading, setLoading] = useState<boolean>(!initialPlan);
  const [error, setError] = useState<string | null>(null);
  
  // Connect to WebSocket for real-time updates
  const { lastMessage } = useWebSocket(executionId);
  
  // Fetch initial plan if not provided
  useEffect(() => {
    const fetchPlan = async () => {
      if (!initialPlan) {
        setLoading(true);
        try {
          const response = await axios.get(`/api/executions/${executionId}/plans`);
          if (response.data && response.data.length > 0) {
            setPlan(response.data[0]);
            
            // Find the active step
            const inProgressIndex = response.data[0].step_statuses.findIndex(
              (status: string) => status === 'in_progress'
            );
            setActiveStep(inProgressIndex !== -1 ? inProgressIndex : 
              response.data[0].step_statuses.filter((status: string) => status === 'completed').length - 1);
          }
        } catch (err) {
          console.error('Error fetching plan:', err);
          setError('Failed to fetch execution plan');
        } finally {
          setLoading(false);
        }
      }
    };
    
    fetchPlan();
  }, [executionId, initialPlan]);
  
  // Process WebSocket messages
  useEffect(() => {
    if (lastMessage && lastMessage.type === 'plan_update' && lastMessage.execution_id === executionId) {
      const updatedPlan = lastMessage.data.plan;
      setPlan(updatedPlan);
      
      // Find the active step
      const inProgressIndex = updatedPlan.step_statuses.findIndex(
        (status: string) => status === 'in_progress'
      );
      setActiveStep(inProgressIndex !== -1 ? inProgressIndex : 
        updatedPlan.step_statuses.filter((status: string) => status === 'completed').length - 1);
    }
  }, [lastMessage, executionId]);
  
  // Toggle step expansion
  const toggleStepExpansion = (stepId: string) => {
    setExpandedSteps(prev => ({
      ...prev,
      [stepId]: !prev[stepId]
    }));
  };
  
  if (loading) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6">Loading plan...</Typography>
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
  
  if (!plan) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6">No plan available</Typography>
          <Typography variant="body2">
            This execution might not use planning, or the plan hasn't been created yet.
          </Typography>
        </CardContent>
      </Card>
    );
  }
  
  // Format steps for visualization
  const formattedSteps = plan.steps.map((step, index) => ({
    id: `step-${index}`,
    title: step,
    description: '',
    status: plan.step_statuses[index] || 'pending',
    notes: plan.step_notes?.[index] || ''
  }));
  
  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">{plan.title || 'Execution Plan'}</Typography>
          <Chip 
            label={`${Math.round(plan.progress || 0)}%`} 
            color="primary" 
            variant="outlined" 
          />
        </Box>
        
        <LinearProgress 
          variant="determinate" 
          value={plan.progress || 0} 
          sx={{ mb: 3, height: 8, borderRadius: 4 }} 
        />
        
        <Stepper activeStep={activeStep} orientation="vertical">
          {formattedSteps.map((step, index) => (
            <Step key={step.id} expanded>
              <StepLabel 
                StepIconComponent={() => <StatusIcon status={step.status} />}
              >
                <Box sx={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center' }}>
                  <Typography variant="subtitle1">{step.title}</Typography>
                  <IconButton 
                    size="small"
                    onClick={() => toggleStepExpansion(step.id)}
                  >
                    {expandedSteps[step.id] ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                  </IconButton>
                </Box>
              </StepLabel>
              <StepContent>
                <Collapse in={expandedSteps[step.id]}>
                  {step.notes && (
                    <Box sx={{ mb: 2, p: 1, bgcolor: 'background.paper', borderRadius: 1 }}>
                      <Typography variant="body2">{step.notes}</Typography>
                    </Box>
                  )}
                  
                  {step.status === 'in_progress' && (
                    <Box sx={{ mt: 1 }}>
                      <LinearProgress />
                    </Box>
                  )}
                </Collapse>
              </StepContent>
            </Step>
          ))}
        </Stepper>
        
        <Box sx={{ mt: 3, display: 'flex', justifyContent: 'space-between' }}>
          <Typography variant="caption">
            Created: {new Date(plan.created_at).toLocaleString()}
          </Typography>
          <Typography variant="caption">
            Last updated: {new Date(plan.updated_at).toLocaleString()}
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default PlanVisualizer;
