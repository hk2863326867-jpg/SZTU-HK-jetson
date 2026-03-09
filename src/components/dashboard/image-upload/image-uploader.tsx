'use client';

import * as React from 'react';
import { useState, useRef } from 'react';
import Button from '@mui/material/Button';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import Paper from '@mui/material/Paper';
import IconButton from '@mui/material/IconButton';
import DeleteIcon from '@mui/icons-material/Delete';
import TextField from '@mui/material/TextField';
import MenuItem from '@mui/material/MenuItem';

const IP_PRESETS = [
  { value: '192.168.55.1', label: 'USB (192.168.55.1)' },
  { value: '10.42.0.1', label: 'WiFi Hotspot (10.42.0.1)' },
  { value: 'custom', label: 'Custom IP' },
];

interface ImageUploaderProps {}

export function ImageUploader({}: ImageUploaderProps): React.JSX.Element {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'success' | 'error' | 'connecting'>('idle');
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [jetsonIp, setJetsonIp] = useState<string>('192.168.55.1');
  const [ipMode, setIpMode] = useState<string>('192.168.55.1');
  const [customIp, setCustomIp] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleIpModeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = event.target.value;
    setIpMode(value);
    if (value !== 'custom') {
      setJetsonIp(value);
    }
  };

  const handleCustomIpChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setCustomIp(event.target.value);
    setJetsonIp(event.target.value);
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setUploadStatus('idle');
      setErrorMessage('');
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    if (!jetsonIp || jetsonIp.trim() === '') {
      setUploadStatus('error');
      setErrorMessage('Please enter Jetson IP address');
      return;
    }

    setIsUploading(true);
    setUploadStatus('connecting');
    setErrorMessage('');

    try {
      const formData = new FormData();
      formData.append('image', selectedFile);
      formData.append('jetsonIp', jetsonIp);

      console.log(`Uploading image to Jetson at ${jetsonIp}...`);

      const response = await fetch('/api/upload-to-jetson', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (response.ok && result.success) {
        setUploadStatus('success');
        console.log('Upload successful:', result);
      } else {
        setUploadStatus('error');
        setErrorMessage(result.error || 'Upload failed');
        console.error('Upload failed:', result);
      }
    } catch (error) {
      setUploadStatus('error');
      setErrorMessage('Failed to upload image');
      console.error('Upload error:', error);
    } finally {
      setIsUploading(false);
    }
  };

  const handleCancel = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setUploadStatus('idle');
    setErrorMessage('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <Box sx={{ width: '100%' }}>
      {/* IP Selection */}
      <Box sx={{ marginBottom: '24px' }}>
        <Typography variant="subtitle1" gutterBottom>
          Connection Method
        </Typography>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-start', flexDirection: { xs: 'column', sm: 'row' } }}>
          <TextField
            select
            label="Connection Type"
            value={ipMode}
            onChange={handleIpModeChange}
            sx={{ minWidth: 200 }}
            size="small"
          >
            {IP_PRESETS.map((option) => (
              <MenuItem key={option.value} value={option.value}>
                {option.label}
              </MenuItem>
            ))}
          </TextField>
          {ipMode === 'custom' && (
            <TextField
              label="Jetson IP Address"
              value={customIp}
              onChange={handleCustomIpChange}
              placeholder="e.g., 192.168.1.100"
              size="small"
              sx={{ minWidth: 200 }}
            />
          )}
          <Typography variant="body2" color="text.secondary" sx={{ alignSelf: 'center' }}>
            Current: {jetsonIp || 'Not set'}
          </Typography>
        </Box>
      </Box>

      {/* Upload Area */}
      <Box
        sx={{
          border: '2px dashed',
          borderColor: 'primary.light',
          borderRadius: '8px',
          padding: '24px',
          textAlign: 'center',
          marginBottom: '24px',
          cursor: 'pointer',
          '&:hover': {
            borderColor: 'primary.main',
            backgroundColor: 'action.hover',
          },
        }}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
        <Typography variant="h6" gutterBottom>
          {selectedFile ? 'Click to change image' : 'Click to select image'}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {selectedFile ? selectedFile.name : 'Supports JPG, PNG, GIF'}
        </Typography>
      </Box>

      {/* Preview */}
      {previewUrl && (
        <Box sx={{ marginBottom: '24px' }}>
          <Paper
            sx={{
              p: 1,
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              position: 'relative',
              maxWidth: '400px',
              margin: '0 auto',
            }}
          >
            <img
              src={previewUrl}
              alt="Preview"
              style={{ maxWidth: '100%', maxHeight: '300px', objectFit: 'contain' }}
            />
            <IconButton
              size="small"
              onClick={handleCancel}
              sx={{
                position: 'absolute',
                top: 8,
                right: 8,
                backgroundColor: 'rgba(255, 255, 255, 0.8)',
              }}
            >
              <DeleteIcon fontSize="small" />
            </IconButton>
          </Paper>
        </Box>
      )}

      {/* Buttons */}
      <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
        <Button
          variant="contained"
          color="primary"
          onClick={handleUpload}
          disabled={!selectedFile || isUploading}
          startIcon={isUploading ? <CircularProgress size={20} /> : undefined}
        >
          {isUploading ? 'Uploading...' : 'Upload to Jetson'}
        </Button>
        {selectedFile && (
          <Button variant="outlined" onClick={handleCancel} disabled={isUploading}>
            Cancel
          </Button>
        )}
      </Box>

      {/* Status */}
      {uploadStatus === 'connecting' && (
        <Alert severity="info" sx={{ mt: 2 }}>
          Connecting to Jetson at {jetsonIp}...
        </Alert>
      )}
      {uploadStatus === 'success' && (
        <Alert severity="success" sx={{ mt: 2 }}>
          Image uploaded successfully!
        </Alert>
      )}
      {uploadStatus === 'error' && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {errorMessage || 'Failed to upload image'}
        </Alert>
      )}
    </Box>
  );
}
