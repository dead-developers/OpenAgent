import axios from 'axios';

// Configure axios defaults
axios.defaults.baseURL = '/api';

// Add request interceptor to include auth token
axios.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Add response interceptor to handle common errors
axios.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle 401 Unauthorized errors
    if (error.response && error.response.status === 401) {
      // Clear token and redirect to login if not already there
      if (window.location.pathname !== '/login') {
        localStorage.removeItem('token');
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

// API functions for executions
export const executionsApi = {
  create: (data: { prompt: string; use_planning: boolean; agent_type: string }) => 
    axios.post('/executions', data),
  
  getById: (id: string) => 
    axios.get(`/executions/${id}`),
  
  getAll: (params?: { skip?: number; limit?: number; status?: string }) => 
    axios.get('/executions', { params }),
  
  getSteps: (executionId: string) => 
    axios.get(`/executions/${executionId}/steps`),
  
  getPlans: (executionId: string) => 
    axios.get(`/executions/${executionId}/plans`)
};

// API functions for tools
export const toolsApi = {
  getExecutions: (params?: { skip?: number; limit?: number; tool_name?: string; status?: string }) => 
    axios.get('/tools/executions', { params }),
  
  getStats: () => 
    axios.get('/tools/stats')
};

// API functions for configuration
export const configApi = {
  getConfig: () => 
    axios.get('/config'),
  
  updateConfig: (data: any) => 
    axios.post('/config', data),
  
  getPresets: () => 
    axios.get('/config/presets'),
  
  getPresetById: (id: string) => 
    axios.get(`/config/presets/${id}`),
  
  createPreset: (data: any) => 
    axios.post('/config/presets', data),
  
  updatePreset: (id: string, data: any) => 
    axios.put(`/config/presets/${id}`, data),
  
  deletePreset: (id: string) => 
    axios.delete(`/config/presets/${id}`)
};

// API functions for authentication
export const authApi = {
  login: (email: string, password: string) => 
    axios.post('/auth/token', 
      new URLSearchParams({
        'username': email,
        'password': password
      }),
      {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        }
      }
    ),
  
  register: (email: string, password: string, fullName: string) => 
    axios.post('/auth/register', {
      email,
      password,
      full_name: fullName
    }),
  
  getCurrentUser: () => 
    axios.get('/auth/me'),
  
  updateProfile: (data: any) => 
    axios.put('/auth/me', data)
};

export default {
  executions: executionsApi,
  tools: toolsApi,
  config: configApi,
  auth: authApi
};
