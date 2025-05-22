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
  Tooltip
} from '@mui/material';
import {
  Search as SearchIcon,
  Refresh as RefreshIcon,
  Visibility as VisibilityIcon,
  Delete as DeleteIcon
} from '@mui/icons-material';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

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

interface ExecutionListResponse {
  executions: Execution[];
  total: number;
  skip: number;
  limit: number;
}

const History: React.FC = () => {
  const [executions, setExecutions] = useState<Execution[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [totalCount, setTotalCount] = useState(0);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [selectedExecution, setSelectedExecution] = useState<Execution | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const navigate = useNavigate();

  // Fetch executions
  const fetchExecutions = async () => {
    setLoading(true);
    setError(null);

    try {
      // Build query parameters
      const params = new URLSearchParams();
      params.append('skip', String(page * rowsPerPage));
      params.append('limit', String(rowsPerPage));

      if (statusFilter) {
        params.append('status', statusFilter);
      }

      // TODO: Add search functionality in the backend
      // if (searchTerm) {
      //   params.append('search', searchTerm);
      // }

      const response = await axios.get<ExecutionListResponse>(`/api/executions?${params.toString()}`);
      
      // Filter by search term client-side for now
      let filteredExecutions = response.data.executions;
      if (searchTerm) {
        filteredExecutions = filteredExecutions.filter(execution => 
          execution.prompt.toLowerCase().includes(searchTerm.toLowerCase()) ||
          execution.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
          (execution.result && execution.result.toLowerCase().includes(searchTerm.toLowerCase()))
        );
      }
      
      setExecutions(filteredExecutions);
      setTotalCount(response.data.total);
    } catch (err) {
      console.error('Error fetching executions:', err);
      setError('Failed to fetch execution history');
    } finally {
      setLoading(false);
    }
  };

  // Fetch executions on mount and when filters change
  useEffect(() => {
    fetchExecutions();
  }, [page, rowsPerPage, statusFilter]);

  // Handle page change
  const handleChangePage = (event: unknown, newPage: number) => {
    setPage(newPage);
  };

  // Handle rows per page change
  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  // Handle search
  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(0);
    fetchExecutions();
  };

  // Handle view details
  const handleViewDetails = (execution: Execution) => {
    setSelectedExecution(execution);
    setDetailsOpen(true);
  };

  // Handle close details
  const handleCloseDetails = () => {
    setDetailsOpen(false);
  };

  // Handle view execution
  const handleViewExecution = (executionId: string) => {
    // Navigate to dashboard with execution ID
    navigate(`/dashboard?execution=${executionId}`);
  };

  // Get status color
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'running':
        return 'primary';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
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
        Execution History
      </Typography>

      {/* Search and filters */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <form onSubmit={handleSearch}>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} sm={6} md={4}>
              <TextField
                fullWidth
                label="Search"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Search by prompt or result"
                variant="outlined"
                size="small"
              />
            </Grid>

            <Grid item xs={12} sm={4} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Status</InputLabel>
                <Select
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                  label="Status"
                >
                  <MenuItem value="">All</MenuItem>
                  <MenuItem value="running">Running</MenuItem>
                  <MenuItem value="completed">Completed</MenuItem>
                  <MenuItem value="error">Error</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} sm={2} md={5}>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Button
                  type="submit"
                  variant="contained"
                  color="primary"
                  startIcon={<SearchIcon />}
                >
                  Search
                </Button>
                <IconButton color="secondary" onClick={fetchExecutions}>
                  <RefreshIcon />
                </IconButton>
              </Box>
            </Grid>
          </Grid>
        </form>
      </Paper>

      {/* Error message */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Executions table */}
      <Paper sx={{ width: '100%', overflow: 'hidden' }}>
        <TableContainer sx={{ maxHeight: 440 }}>
          <Table stickyHeader aria-label="executions table">
            <TableHead>
              <TableRow>
                <TableCell>ID</TableCell>
                <TableCell>Prompt</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Agent Type</TableCell>
                <TableCell>Start Time</TableCell>
                <TableCell>End Time</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {loading ? (
                <TableRow>
                  <TableCell colSpan={7} align="center">
                    <CircularProgress size={24} sx={{ my: 2 }} />
                  </TableCell>
                </TableRow>
              ) : executions.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} align="center">
                    No executions found
                  </TableCell>
                </TableRow>
              ) : (
                executions.map((execution) => (
                  <TableRow key={execution.id} hover>
                    <TableCell>
                      <Tooltip title={execution.id}>
                        <span>{truncateText(execution.id, 8)}</span>
                      </Tooltip>
                    </TableCell>
                    <TableCell>
                      <Tooltip title={execution.prompt}>
                        <span>{truncateText(execution.prompt, 50)}</span>
                      </Tooltip>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={execution.status}
                        color={getStatusColor(execution.status) as any}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>{execution.agent_type || 'N/A'}</TableCell>
                    <TableCell>{formatDate(execution.start_time)}</TableCell>
                    <TableCell>
                      {execution.end_time ? formatDate(execution.end_time) : 'N/A'}
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', gap: 1 }}>
                        <Tooltip title="View Details">
                          <IconButton
                            size="small"
                            onClick={() => handleViewDetails(execution)}
                          >
                            <VisibilityIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Open in Dashboard">
                          <IconButton
                            size="small"
                            onClick={() => handleViewExecution(execution.id)}
                            color="primary"
                          >
                            <VisibilityIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </Box>
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
          count={totalCount}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={handleChangePage}
          onRowsPerPageChange={handleChangeRowsPerPage}
        />
      </Paper>

      {/* Execution details dialog */}
      <Dialog
        open={detailsOpen}
        onClose={handleCloseDetails}
        maxWidth="md"
        fullWidth
      >
        {selectedExecution && (
          <>
            <DialogTitle>
              Execution Details
              <Chip
                label={selectedExecution.status}
                color={getStatusColor(selectedExecution.status) as any}
                size="small"
                sx={{ ml: 1 }}
              />
            </DialogTitle>
            <DialogContent dividers>
              <Typography variant="subtitle1" gutterBottom>
                ID: {selectedExecution.id}
              </Typography>
              <Typography variant="subtitle2" gutterBottom>
                Agent Type: {selectedExecution.agent_type || 'N/A'}
              </Typography>
              <Typography variant="subtitle2" gutterBottom>
                Start Time: {formatDate(selectedExecution.start_time)}
              </Typography>
              {selectedExecution.end_time && (
                <Typography variant="subtitle2" gutterBottom>
                  End Time: {formatDate(selectedExecution.end_time)}
                </Typography>
              )}
              
              <Divider sx={{ my: 2 }} />
              
              <Typography variant="h6" gutterBottom>
                Prompt
              </Typography>
              <Paper sx={{ p: 2, bgcolor: 'background.default', mb: 2 }}>
                <Typography variant="body1" component="pre" sx={{ whiteSpace: 'pre-wrap' }}>
                  {selectedExecution.prompt}
                </Typography>
              </Paper>
              
              {selectedExecution.result && (
                <>
                  <Typography variant="h6" gutterBottom>
                    Result
                  </Typography>
                  <Paper sx={{ p: 2, bgcolor: 'background.default' }}>
                    <Typography variant="body1" component="pre" sx={{ whiteSpace: 'pre-wrap' }}>
                      {selectedExecution.result}
                    </Typography>
                  </Paper>
                </>
              )}
            </DialogContent>
            <DialogActions>
              <Button onClick={handleCloseDetails}>Close</Button>
              <Button 
                variant="contained" 
                color="primary"
                onClick={() => {
                  handleCloseDetails();
                  handleViewExecution(selectedExecution.id);
                }}
              >
                Open in Dashboard
              </Button>
            </DialogActions>
          </>
        )}
      </Dialog>
    </Container>
  );
};

export default History;
