import React, { useState, useEffect } from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Chip,
  TextField,
  IconButton,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  CircularProgress,
  Alert,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Tooltip,
  Card,
  CardContent,
  Divider,
  LinearProgress
} from '@mui/material';
import {
  Search as SearchIcon,
  Refresh as RefreshIcon,
  Visibility as VisibilityIcon,
  Code as CodeIcon,
  Timeline as TimelineIcon
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

interface ToolStats {
  tool_name: string;
  count: number;
  avg_execution_time: number;
  success_rate: number;
  last_execution: string;
}

interface ToolMonitoringState {
  executions: ToolExecution[];
  stats: ToolStats[];
  loading: boolean;
  error: string | null;
  page: number;
  rowsPerPage: number;
  totalCount: number;
  searchTerm: string;
  toolFilter: string;
  statusFilter: string;
  selectedExecution: ToolExecution | null;
  detailsOpen: boolean;
}

const ToolMonitoring: React.FC = () => {
  const [state, setState] = useState<ToolMonitoringState>({
    executions: [],
    stats: [],
    loading: true,
    error: null,
    page: 0,
    rowsPerPage: 10,
    totalCount: 0,
    searchTerm: '',
    toolFilter: '',
    statusFilter: '',
    selectedExecution: null,
    detailsOpen: false
  });

  // Connect to WebSocket for real-time updates
  const { lastMessage } = useWebSocket('tool-monitoring');

  // Fetch tool executions
  const fetchToolExecutions = async () => {
    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      // Build query parameters
      const params = new URLSearchParams();
      params.append('skip', String(state.page * state.rowsPerPage));
      params.append('limit', String(state.rowsPerPage));

      if (state.toolFilter) {
        params.append('tool_name', state.toolFilter);
      }

      if (state.statusFilter) {
        params.append('status', state.statusFilter);
      }

      const response = await axios.get(`/api/tools/executions?${params.toString()}`);
      
      // Filter by search term client-side for now
      let filteredExecutions = response.data.executions;
      if (state.searchTerm) {
        filteredExecutions = filteredExecutions.filter((execution: ToolExecution) => 
          execution.tool_name.toLowerCase().includes(state.searchTerm.toLowerCase()) ||
          execution.id.toLowerCase().includes(state.searchTerm.toLowerCase()) ||
          execution.execution_id.toLowerCase().includes(state.searchTerm.toLowerCase())
        );
      }
      
      setState(prev => ({
        ...prev,
        executions: filteredExecutions,
        totalCount: response.data.total,
        loading: false
      }));
    } catch (err) {
      console.error('Error fetching tool executions:', err);
      setState(prev => ({
        ...prev,
        error: 'Failed to fetch tool executions',
        loading: false
      }));
    }
  };

  // Fetch tool stats
  const fetchToolStats = async () => {
    try {
      const response = await axios.get('/api/tools/stats');
      setState(prev => ({
        ...prev,
        stats: response.data.stats
      }));
    } catch (err) {
      console.error('Error fetching tool stats:', err);
      // Don't set error state here to avoid overriding execution errors
    }
  };

  // Fetch data on mount and when filters change
  useEffect(() => {
    fetchToolExecutions();
    fetchToolStats();
  }, [state.page, state.rowsPerPage, state.toolFilter, state.statusFilter]);

  // Process WebSocket messages
  useEffect(() => {
    if (lastMessage && lastMessage.type === 'tool_execution') {
      const newExecution = lastMessage.data.execution;
      
      // Add to executions if on first page
      if (state.page === 0) {
        setState(prev => ({
          ...prev,
          executions: [newExecution, ...prev.executions.slice(0, prev.rowsPerPage - 1)],
          totalCount: prev.totalCount + 1
        }));
      } else {
        // Just update the count
        setState(prev => ({
          ...prev,
          totalCount: prev.totalCount + 1
        }));
      }
      
      // Refresh stats
      fetchToolStats();
    }
  }, [lastMessage]);

  // Handle page change
  const handleChangePage = (event: unknown, newPage: number) => {
    setState(prev => ({ ...prev, page: newPage }));
  };

  // Handle rows per page change
  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setState(prev => ({
      ...prev,
      rowsPerPage: parseInt(event.target.value, 10),
      page: 0
    }));
  };

  // Handle search
  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setState(prev => ({ ...prev, page: 0 }));
    fetchToolExecutions();
  };

  // Handle view details
  const handleViewDetails = (execution: ToolExecution) => {
    setState(prev => ({
      ...prev,
      selectedExecution: execution,
      detailsOpen: true
    }));
  };

  // Handle close details
  const handleCloseDetails = () => {
    setState(prev => ({ ...prev, detailsOpen: false }));
  };

  // Format date
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  // Truncate text
  const truncateText = (text: string, maxLength: number) => {
    if (!text) return '';
    return text.length > maxLength ? `${text.substring(0, maxLength)}...` : text;
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        Tool Monitoring
      </Typography>

      {/* Tool Stats */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {state.stats.slice(0, 4).map((stat) => (
          <Grid item xs={12} sm={6} md={3} key={stat.tool_name}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom noWrap>
                  {stat.tool_name}
                </Typography>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2" color="text.secondary">
                    Executions:
                  </Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {stat.count}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2" color="text.secondary">
                    Avg Time:
                  </Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {stat.avg_execution_time.toFixed(2)}ms
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    Success Rate:
                  </Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {(stat.success_rate * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <LinearProgress 
                  variant="determinate" 
                  value={stat.success_rate * 100} 
                  color={stat.success_rate > 0.9 ? "success" : stat.success_rate > 0.7 ? "primary" : "error"}
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Search and filters */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <form onSubmit={handleSearch}>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} sm={6} md={4}>
              <TextField
                fullWidth
                label="Search"
                value={state.searchTerm}
                onChange={(e) => setState(prev => ({ ...prev, searchTerm: e.target.value }))}
                placeholder="Search by tool name or ID"
                variant="outlined"
                size="small"
              />
            </Grid>

            <Grid item xs={12} sm={3} md={2}>
              <FormControl fullWidth size="small">
                <InputLabel>Tool</InputLabel>
                <Select
                  value={state.toolFilter}
                  onChange={(e) => setState(prev => ({ ...prev, toolFilter: e.target.value }))}
                  label="Tool"
                >
                  <MenuItem value="">All Tools</MenuItem>
                  {state.stats.map(stat => (
                    <MenuItem key={stat.tool_name} value={stat.tool_name}>
                      {stat.tool_name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} sm={3} md={2}>
              <FormControl fullWidth size="small">
                <InputLabel>Status</InputLabel>
                <Select
                  value={state.statusFilter}
                  onChange={(e) => setState(prev => ({ ...prev, statusFilter: e.target.value }))}
                  label="Status"
                >
                  <MenuItem value="">All</MenuItem>
                  <MenuItem value="success">Success</MenuItem>
                  <MenuItem value="error">Error</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} sm={12} md={4}>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Button
                  type="submit"
                  variant="contained"
                  color="primary"
                  startIcon={<SearchIcon />}
                >
                  Search
                </Button>
                <IconButton color="secondary" onClick={() => {
                  fetchToolExecutions();
                  fetchToolStats();
                }}>
                  <RefreshIcon />
                </IconButton>
              </Box>
            </Grid>
          </Grid>
        </form>
      </Paper>

      {/* Error message */}
      {state.error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {state.error}
        </Alert>
      )}

      {/* Executions table */}
      <Paper sx={{ width: '100%', overflow: 'hidden' }}>
        <TableContainer sx={{ maxHeight: 440 }}>
          <Table stickyHeader aria-label="tool executions table">
            <TableHead>
              <TableRow>
                <TableCell>Tool</TableCell>
                <TableCell>Execution ID</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Execution Time</TableCell>
                <TableCell>Timestamp</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {state.loading ? (
                <TableRow>
                  <TableCell colSpan={6} align="center">
                    <CircularProgress size={24} sx={{ my: 2 }} />
                  </TableCell>
                </TableRow>
              ) : state.executions.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={6} align="center">
                    No tool executions found
                  </TableCell>
                </TableRow>
              ) : (
                state.executions.map((execution) => (
                  <TableRow key={execution.id} hover>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <CodeIcon sx={{ mr: 1, fontSize: 16 }} />
                        {execution.tool_name}
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Tooltip title={execution.execution_id}>
                        <span>{truncateText(execution.execution_id, 8)}</span>
                      </Tooltip>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={execution.status}
                        color={execution.status === 'success' ? 'success' : 'error'}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>{execution.execution_time}ms</TableCell>
                    <TableCell>{formatDate(execution.timestamp)}</TableCell>
                    <TableCell>
                      <IconButton
                        size="small"
                        onClick={() => handleViewDetails(execution)}
                      >
                        <VisibilityIcon fontSize="small" />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </TableContainer>
        <TablePagination
          rowsPerPageOptions={[5, 10, 25, 50]}
          component="div"
          count={state.totalCount}
          rowsPerPage={state.rowsPerPage}
          page={state.page}
          onPageChange={handleChangePage}
          onRowsPerPageChange={handleChangeRowsPerPage}
        />
      </Paper>

      {/* Execution details dialog */}
      <Dialog
        open={state.detailsOpen}
        onClose={handleCloseDetails}
        maxWidth="md"
        fullWidth
      >
        {state.selectedExecution && (
          <>
            <DialogTitle>
              Tool Execution Details
              <Chip
                label={state.selectedExecution.status}
                color={state.selectedExecution.status === 'success' ? 'success' : 'error'}
                size="small"
                sx={{ ml: 1 }}
              />
            </DialogTitle>
            <DialogContent dividers>
              <Typography variant="subtitle1" gutterBottom>
                Tool: {state.selectedExecution.tool_name}
              </Typography>
              <Typography variant="subtitle2" gutterBottom>
                Execution ID: {state.selectedExecution.execution_id}
              </Typography>
              <Typography variant="subtitle2" gutterBottom>
                Step Number: {state.selectedExecution.step_number}
              </Typography>
              <Typography variant="subtitle2" gutterBottom>
                Execution Time: {state.selectedExecution.execution_time}ms
              </Typography>
              <Typography variant="subtitle2" gutterBottom>
                Timestamp: {formatDate(state.selectedExecution.timestamp)}
              </Typography>
              
              <Divider sx={{ my: 2 }} />
              
              <Typography variant="h6" gutterBottom>
                Input Data
              </Typography>
              <Paper sx={{ p: 2, bgcolor: 'background.default', mb: 2 }}>
                <Typography variant="body1" component="pre" sx={{ whiteSpace: 'pre-wrap' }}>
                  {JSON.stringify(state.selectedExecution.input_data, null, 2)}
                </Typography>
              </Paper>
              
              <Typography variant="h6" gutterBottom>
                Output Data
              </Typography>
              <Paper sx={{ p: 2, bgcolor: 'background.default' }}>
                <Typography variant="body1" component="pre" sx={{ whiteSpace: 'pre-wrap' }}>
                  {JSON.stringify(state.selectedExecution.output_data, null, 2)}
                </Typography>
              </Paper>
            </DialogContent>
            <DialogActions>
              <Button onClick={handleCloseDetails}>Close</Button>
            </DialogActions>
          </>
        )}
      </Dialog>
    </Container>
  );
};

export default ToolMonitoring;
