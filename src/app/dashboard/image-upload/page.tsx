import * as React from 'react';
import type { Metadata } from 'next';
import Grid from '@mui/material/Grid';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardHeader from '@mui/material/CardHeader';
import Typography from '@mui/material/Typography';

import { config } from '@/config';

// 导入图片上传组件
import { ImageUploader } from '@/components/dashboard/image-upload/image-uploader';

export const metadata = {
  title: `Image Upload | Dashboard | ${config.site.name}`,
} satisfies Metadata;

export default function Page(): React.JSX.Element {
  return (
    <Grid container spacing={3}>
      <Grid size={{ lg: 12, xs: 12 }}>
        <Card>
          <CardHeader title="Image Upload" subtitle="Upload images to Jetson device" />
          <CardContent>
            <ImageUploader />
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
}
