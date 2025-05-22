import React, { useState, useEffect } from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Divider,
  Alert,
  Snackbar,
  IconButton,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Tooltip,
  CircularProgress,
  Chip
} from '@mui/material';
import {
  Save as SaveIcon,
  Refresh as RefreshIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  ExpandMore as ExpandMoreIcon,
  HelpOutline as HelpIcon,
  ContentCopy as CopyIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon
} from '@mui/icons-material';
import axios from 'axios';

// Types
interface ConfigField {
  id: string;
  label: string;
  type: 'text' | 'number' | 'boolean' | 'select' | 'password';
  description?: string;
  options?: { value: string; label: string }[];
  value: any;
  required?: boolean;
  validation?: {
    pattern?: string;
    min?: number;
    max?: number;
    message?: string;
  };
}

interface ConfigSection {
  id: string;
  title: string;
  description: string;
  fields: ConfigField[];
}

interface ConfigPreset {
  id: string;
  name: string;
  description: string;
  is_default: boolean;
  created_at: string;
  updated_at: string;
}

const Settings: React.FC = () => {
  const [configSections, setConfigSections] = useState<ConfigSection[]>([]);
  const [presets, setPresets] = useState<ConfigPreset[]>([]);
  const [activePreset, setActivePreset] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [showSecrets, setShowSecrets] = useState<Record<string, boolean>>({});
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({});
  const [newPresetName, setNewPresetName] = useState('');
  const [newPresetDescription, setNewPresetDescription] = useState('');
  const [showNewPresetDialog, setShowNewPresetDialog] = useState(false);

  // Fetch configuration
  const fetchConfiguration = async () => {
    setLoading(true);
    setError(null);

    try {
      // Fetch config sections
      const response = await axios.get('/api/config');
      setConfigSections(response.data.sections);

      // Fetch presets
      const presetsResponse = await axios.get('/api/config/presets');
      setPresets(presetsResponse.data.presets);

      // Set active preset
      const defaultPreset = presetsResponse.data.presets.find((p: ConfigPreset) => p.is_default);
      if (defaultPreset) {
        setActivePreset(defaultPreset.id);
      }
    } catch (err) {
      console.error('Error fetching configuration:', err);
      setError('Failed to fetch configuration. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Initial fetch
  useEffect(() => {
    fetchConfiguration();
  }, []);

  // Load preset
  const loadPreset = async (presetId: string) => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.get(`/api/config/presets/${presetId}`);
      setConfigSections(response.data.sections);
      setActivePreset(presetId);
      setSuccess('Preset loaded successfully');
    } catch (err) {
      console.error('Error loading preset:', err);
      setError('Failed to load preset. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Save configuration
  const saveConfiguration = async () => {
    // Validate all fields
    const errors: Record<string, string> = {};

    configSections.forEach(section => {
      section.fields.forEach(field => {
        // Check required fields
        if (field.required && !field.value) {
          errors[`${section.id}.${field.id}`] = 'This field is required';
          return;
        }

        // Check validation rules
        if (field.validation) {
          // Pattern validation
          if (field.validation.pattern && field.type === 'text') {
            const regex = new RegExp(field.validation.pattern);
            if (!regex.test(field.value)) {
              errors[`${section.id}.${field.id}`] = field.validation.message || 'Invalid format';
              return;
            }
          }

          // Number range validation
          if ((field.validation.min !== undefined || field.validation.max !== undefined) &&
            field.type === 'number') {
            const numValue = Number(field.value);
            if (field.validation.min !== undefined && numValue < field.validation.min) {
              errors[`${section.id}.${field.id}`] = `Value must be at least ${field.validation.min}`;
              return;
            }
            if (field.validation.max !== undefined && numValue > field.validation.max) {
              errors[`${section.id}.${field.id}`] = `Value must be at most ${field.validation.max}`;
              return;
            }
          }
        }
      });
    });

    // If there are validation errors, show them and don't save
    if (Object.keys(errors).length > 0) {
      setValidationErrors(errors);
      setError('Please fix the validation errors before saving');
      return;
    }

    // Clear validation errors
    setValidationErrors({});

    // Save configuration
    setSaving(true);
    setError(null);

    try {
      await axios.post('/api/config', { sections: configSections });
      setSuccess('Configuration saved successfully');

      // Refresh presets
      const presetsResponse = await axios.get('/api/config/presets');
      setPresets(presetsResponse.data.presets);
    } catch (err) {
      console.error('Error saving configuration:', err);
      setError('Failed to save configuration. Please try again.');
    } finally {
      setSaving(false);
    }
  };

  // Save as new preset
  const saveAsNewPreset = async () => {
    if (!newPresetName) {
      setError('Preset name is required');
      return;
    }

    setSaving(true);
    setError(null);

    try {
      const response = await axios.post('/api/config/presets', {
        name: newPresetName,
        description: newPresetDescription,
        sections: configSections
      });

      setSuccess('Preset created successfully');
      setActivePreset(response.data.id);
      setNewPresetName('');
      setNewPresetDescription('');
      setShowNewPresetDialog(false);

      // Refresh presets
      const presetsResponse = await axios.get('/api/config/presets');
      setPresets(presetsResponse.data.presets);
    } catch (err) {
      console.error('Error creating preset:', err);
      setError('Failed to create preset. Please try again.');
    } finally {
      setSaving(false);
    }
  };

  // Handle field change
  const handleFieldChange = (sectionId: string, fieldId: string, value: any) => {
    setConfigSections(prevSections =>
      prevSections.map(section =>
        section.id === sectionId
          ? {
            ...section,
            fields: section.fields.map(field =>
              field.id === fieldId
                ? { ...field, value }
                : field
            )
          }
          : section
      )
    );

    // Clear validation error if exists
    if (validationErrors[`${sectionId}.${fieldId}`]) {
      setValidationErrors(prev => {
        const newErrors = { ...prev };
        delete newErrors[`${sectionId}.${fieldId}`];
        return newErrors;
      });
    }
  };

  // Toggle password visibility
  const togglePasswordVisibility = (fieldId: string) => {
    setShowSecrets(prev => ({
      ...prev,
      [fieldId]: !prev[fieldId]
    }));
  };

  // Render field based on type
  const renderField = (section: ConfigSection, field: ConfigField) => {
    const fieldKey = `${section.id}.${field.id}`;
    const hasError = !!validationErrors[fieldKey];

    switch (field.type) {
      case 'text':
      case 'number':
        return (
          <TextField
            fullWidth
            label={field.label}
            type={field.type === 'number' ? 'number' : 'text'}
            value={field.value || ''}
            onChange={(e) => handleFieldChange(section.id, field.id, e.target.value)}
            error={hasError}
            helperText={validationErrors[fieldKey] || field.description}
            required={field.required}
            margin="normal"
            size="small"
          />
        );

      case 'password':
        return (
          <TextField
            fullWidth
            label={field.label}
            type={showSecrets[field.id] ? 'text' : 'password'}
            value={field.value || ''}
            onChange={(e) => handleFieldChange(section.id, field.id, e.target.value)}
            error={hasError}
            helperText={validationErrors[fieldKey] || field.description}
            required={field.required}
            margin="normal"
            size="small"
            InputProps={{
              endAdornment: (
                <IconButton
                  onClick={() => togglePasswordVisibility(field.id)}
                  edge="end"
                >
                  {showSecrets[field.id] ? <VisibilityOffIcon /> : <VisibilityIcon />}
                </IconButton>
              )
            }}
          />
        );

      case 'boolean':
        return (
          <FormControlLabel
            control={
              <Switch
                checked={!!field.value}
                onChange={(e) => handleFieldChange(section.id, field.id, e.target.checked)}
                color="primary"
              />
            }
            label={field.label}
          />
        );

      case 'select':
        return (
          <FormControl fullWidth margin="normal" size="small" error={hasError}>
            <InputLabel>{field.label}</InputLabel>
            <Select
              value={field.value || ''}
              onChange={(e) => handleFieldChange(section.id, field.id, e.target.value)}
              label={field.label}
            >
              {field.options?.map(option => (
                <MenuItem key={option.value} value={option.value}>
                  {option.label}
                </MenuItem>
              ))}
            </Select>
            {(validationErrors[fieldKey] || field.description) && (
              <Typography variant="caption" color={hasError ? "error" : "text.secondary"}>
                {validationErrors[fieldKey] || field.description}
              </Typography>
            )}
          </FormControl>
        );

      default:
        return null;
    }
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        Configuration Settings
      </Typography>

      {/* Presets selector */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} sm={4}>
            <FormControl fullWidth size="small">
              <InputLabel>Configuration Preset</InputLabel>
              <Select
                value={activePreset}
                onChange={(e) => loadPreset(e.target.value as string)}
                label="Configuration Preset"
                disabled={loading || saving}
              >
                {presets.map(preset => (
                  <MenuItem key={preset.id} value={preset.id}>
                    {preset.name}
                    {preset.is_default && (
                      <Chip
                        label="Default"
                        size="small"
                        color="primary"
                        sx={{ ml: 1 }}
                      />
                    )}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} sm={8}>
            <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
              <Button
                variant="outlined"
                startIcon={<AddIcon />}
                onClick={() => setShowNewPresetDialog(true)}
                disabled={loading || saving}
              >
                Save as New Preset
              </Button>

              <Button
                variant="contained"
                color="primary"
                startIcon={<SaveIcon />}
                onClick={saveConfiguration}
                disabled={loading || saving}
              >
                {saving ? 'Saving...' : 'Save Configuration'}
              </Button>

              <IconButton
                color="secondary"
                onClick={fetchConfiguration}
                disabled={loading || saving}
              >
                <RefreshIcon />
              </IconButton>
            </Box>
          </Grid>
        </Grid>
      </Paper>

      {/* New preset dialog */}
      {showNewPresetDialog && (
        <Paper sx={{ p: 2, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Create New Preset
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Preset Name"
                value={newPresetName}
                onChange={(e) => setNewPresetName(e.target.value)}
                required
                margin="normal"
                size="small"
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Description"
                value={newPresetDescription}
                onChange={(e) => setNewPresetDescription(e.target.value)}
                multiline
                rows={2}
                margin="normal"
                size="small"
              />
            </Grid>
            <Grid item xs={12}>
              <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 1 }}>
                <Button
                  variant="outlined"
                  onClick={() => setShowNewPresetDialog(false)}
                >
                  Cancel
                </Button>
                <Button
                  variant="contained"
                  color="primary"
                  onClick={saveAsNewPreset}
                  disabled={!newPresetName || saving}
                >
                  Create Preset
                </Button>
              </Box>
            </Grid>
          </Grid>
        </Paper>
      )}

      {/* Error and success messages */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Snackbar
        open={!!success}
        autoHideDuration={6000}
        onClose={() => setSuccess(null)}
        message={success}
      />

      {/* Configuration sections */}
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      ) : (
        configSections.map(section => (
          <Accordion key={section.id} defaultExpanded sx={{ mb: 2 }}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="h6">{section.title}</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="body2" color="text.secondary" paragraph>
                {section.description}
              </Typography>

              <Grid container spacing={2}>
                {section.fields.map(field => (
                  <Grid item xs={12} sm={6} key={field.id}>
                    {renderField(section, field)}
                  </Grid>
                ))}
              </Grid>
            </AccordionDetails>
          </Accordion>
        ))
      )}
    </Container>
  );
};

export default Settings;
