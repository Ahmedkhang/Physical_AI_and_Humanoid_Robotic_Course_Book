import React from 'react';
import Layout from '@theme/Layout';
import PersonalizationDashboard from '@site/src/components/PersonalizationDashboard';

export default function PersonalizationDashboardPage() {
  return (
    <Layout title="Personalization Dashboard" description="Customize your learning experience">
      <PersonalizationDashboard />
    </Layout>
  );
}